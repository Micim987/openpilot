# Porting JetLink support to NRDR

This is a rebase and forward-port guide for the JetLink integration currently
running in NRDR. It records the working architecture, every integration area,
the NRDR-specific merge decisions, and the validation that was performed. It
is intentionally more useful than a one-time list of copied files: future
versions of NRDR, sunnypilot, ZoomPilot, and JetLink will not have identical
trees or APIs.

## Known-good reference

Hardware validation succeeded on a comma `tizi` with a Jetson on 2026-09-27.
The large model was downloaded, compiled, and used successfully through
JetLink.

| Item | Known-good revision |
| --- | --- |
| NRDR source branch | `nrdr/openpilot:nrdr-architecture-development` |
| NRDR base used by this port | `425a33eaff4a4e00d1641bd6716c64ec8539535b` |
| Working integration branch | `Micim987/openpilot:nrdr-new-jetlink` |
| Initial port commit | `2675be05f` |
| JetLink v0.4/UI follow-up | `138e2d05d` |
| ZoomPilot source branch | `zoompilot/zoompilot:jetson-trt` |
| Initial source snapshot | `9056e9a4f` |
| Source tip checked for the follow-up | `1c85c44cb` |
| `jetlink_repo` submodule | `3e6a59f9fcae28abaf942e66d6e8758afa6103c9` |

The commit IDs are provenance, not an instruction to blindly cherry-pick.
ZoomPilot contains unrelated vehicle, UI, cereal, and controls changes that
must not replace NRDR behavior.

## Required behavior

The port has four non-negotiable properties:

1. JetLink can provision and run the selected big model on an attached Jetson.
2. The local small model remains available while JetLink connects, builds,
   fails, or reconnects.
3. Native Chestnut behavior continues to work and remains distinct from an
   off-board accelerator.
4. NRDR's Honda torque modifications, hooks, events, model selection, settings,
   and control behavior remain intact.

## Architecture

The accelerator interface is deliberately separate from the JetLink backend:

```text
model manager ── big-model catalog/selection ───────────────┐
                                                            │
manager ── owner process ── offroad provisioning worker     │
                 │                     │                    │
                 │                     ├─ download ONNX     │
                 │                     ├─ upload to Jetson  │
                 │                     └─ build TRT engine  │
                 │                                          │
modeld ── joining model state ── JetLink client ── Jetson ◀─┘
   │             │
   │             └─ falls back to the local small model
   └─ modelV2/modelDataV2SP status ── selfdrived + UI
```

`openpilot/sunnypilot/accelerators/__init__.py` is the stable adapter used by
the rest of openpilot. JetLink-specific implementation lives under
`openpilot/sunnypilot/accelerators/jetlink/`. Future ports should keep callers
on the adapter instead of importing JetLink internals.

The long-lived `owner.py` holds the USB gadget while JetLink is enabled. It
lends the endpoints to either the offroad provisioning worker or modeld so the
Jetson does not see an unplug at every ignition transition. The heavyweight
`jetlinkd.py` worker only runs when provisioning work exists.

## Change inventory

### Dependency, build, and boot setup

- `.gitmodules` registers `jetlink_repo` from
  `https://github.com/zoompilot/jetlink.git`.
- `.gitignore` ignores the generated top-level `/jetlink` symlink.
- `launch_chffrplus.sh` creates that symlink and invokes the accelerator setup
  script before manager starts.
- `openpilot/sunnypilot/SConscript` includes
  `accelerators/SConscript`.
- `openpilot/sunnypilot/accelerators/SConscript` builds the camera warp and
  includes the accelerator sources required on-device.
- `openpilot/sunnypilot/accelerators/setup.sh` dispatches backend setup.
- `openpilot/sunnypilot/accelerators/jetlink/setup.sh` creates the AGNOS USB
  gadget only when `JetlinkEnabled` is true. The working v0.4 version reads the
  param file directly because this runs before the Python params library is
  guaranteed to be built.

Always update the submodule pointer and the Python integration together. A
checkout with mismatched revisions may import successfully and then fail when
it encounters a renamed catalog or model-spec API.

### Accelerator and JetLink implementation

The complete `openpilot/sunnypilot/accelerators/` package was brought over,
including its tests. Major responsibilities are:

- `__init__.py`: public accelerator interface, process declaration, progress,
  readiness, shutdown, selected model, and catalog hooks.
- `jetlink/backend.py`: maps the generic interface to JetLink.
- `jetlink/gadget.py`, `owner.py`, and `lending.py`: USB gadget ownership and
  endpoint handoff.
- `jetlink/jetlinkd.py`, `provision.py`, and `lfs.py`: offroad download,
  transfer, and engine provisioning.
- `jetlink/joining.py` and `fallback.py`: asynchronous big-model join, demotion,
  retry, and continued small-model operation.
- `jetlink/model_state.py`, `warp_cache.py`, and `compile_warp.py`: modeld
  adapter and comma-side image warp.
- `jetlink/spec_cache.py` and `status.py`: engine metadata and runtime status.
- `jetlink/vmtune.py`: VM settings required by the accelerator lifecycle.

JetLink v0.4 changed two important interfaces:

- Catalog discovery uses `jetlink.registry.catalog.fetch_catalogs()` rather
  than `newer_catalogs(url)`.
- Model specs expose `packed_layout` and `feed_back()` rather than
  `packed_sizes`, `packed_shapes`, and manual `prev_feat` copying.

When updating JetLink, search both repositories for the old and new API names
before assuming a submodule bump is sufficient.

### Model catalog and model manager

The big-model slot is shared by native Chestnut and JetLink, but their file
requirements differ:

- `models/fetcher.py` passes the Chestnut/big catalog through
  `accelerators.big_catalog()`. With JetLink enabled, the pinned catalog is
  merged with newer catalogs supported by the JetLink registry.
- `models/manager.py` stores the big-model selection without downloading its
  local artifacts when JetLink, rather than native Chestnut, will execute it.
- A physical Chestnut still fetches and validates the local big-model files.
- `models/helpers.py` does not reset a valid big-model selection merely because
  its local files are absent on a JetLink installation.
- Model selection and validation tests were updated for the two execution
  paths.

NRDR-specific rule: retain
`openpilot.nrdr.features.services.model_manager.select_default_model`. Do not
replace NRDR's manager wholesale with ZoomPilot's manager, and do not replace
this call with a fork-specific default-selection implementation without
reviewing NRDR's policy.

### Stock modeld and sunnypilot modeld_v2

Both modeld implementations must support the accelerator:

- `openpilot/selfdrive/modeld/modeld.py`
- `openpilot/sunnypilot/modeld_v2/modeld.py`

Each implementation must:

1. Ask the accelerator to prepare before entering the real-time loop.
2. Create a joining state around the local small model.
3. Pass an `after_enqueue` callback so JetLink health/status can be published
   while the Jetson is performing inference.
4. Publish whether the big model is available, joining, running, retrying, or
   unavailable.
5. Preserve `modelV2.big` semantics so downstream tuning and controls know
   which model is actually executing.
6. Close or demote the remote model cleanly and keep the small model alive on
   link loss.

NRDR's `modeld_v2` call signatures and timing metadata differ from ZoomPilot's
source tree. Adapt the joining state to the local `ModelState` contract; do not
overwrite the entire NRDR model loop. In particular, preserve NRDR's lateral
action timing, live-delay handling, message publication, and any later NRDR
model hooks.

### Selfdrived events and transition policy

The port adds an accelerator-specific adapter at
`openpilot/sunnypilot/selfdrive/selfdrived/accelerator_events.py` and integrates
it through the existing selfdrived event flow.

The relevant states distinguish:

- a big model that is still loading;
- a big model connected and ready for a safe transition;
- a big model that is actively running; and
- a link that was lost while driving.

The transition policy avoids swapping models in an unsafe window and keeps the
small model as the fallback. When merging this area, preserve every NRDR import,
`NrdrSelfdrive` hook, Honda event filter, longitudinal policy, and event update.
Only merge the accelerator state alongside those features.

### Cereal schema, event enums, and params

`openpilot/cereal/custom.capnp` gained:

- `bigModelAvailable` and `bigModelLinkLost` custom onroad events;
- `ModelDataV2SP.bigModelAvailable`;
- `ModelDataV2SP.acceleratorState`; and
- `ModelDataV2SP.acceleratorName`.

Cap'n Proto and event ordinals are compatibility boundaries. On a future
rebase, append or merge fields into the new base's available ordinals. Never
copy the old struct over a newer NRDR struct, renumber existing fields, or
reuse an ordinal that the rebased branch already assigned.

`openpilot/common/params_keys.h` gained:

```text
AcceleratorProgress
Offroad_AcceleratorUnavailable
JetlinkEnabled
JetlinkEndpoint
JetlinkModel                 (legacy migration key)
JetlinkEngineReady
JetlinkSpec
JetlinkCachedModels
JetlinkModelPointers
```

Preserve the attributes shown in the working branch. In particular,
`JetlinkEnabled` and `JetlinkEndpoint` are persistent and backed up, while
engine/spec/cache state has a different lifecycle.

### Process, hardware, and shutdown lifecycle

- `openpilot/system/manager/process_config.py` registers the daemon declared by
  the accelerator adapter. The owner intentionally remains alive both onroad
  and offroad while enabled.
- `openpilot/system/hardware/hardwared.py` publishes the accelerator-unavailable
  offroad alert and asks an externally powered Jetson to shut down with the
  comma.
- `openpilot/selfdrive/selfdrived/alerts_offroad.json` defines the user-facing
  unavailable alert.

Do not gate the owner as a conventional offroad-only process. The provisioning
worker is offroad-only; the lightweight owner must survive the ignition edge
to preserve the USB gadget and lend it to modeld.

### UI and settings

The port added or updated:

- `openpilot/selfdrive/ui/sunnypilot/accelerator_link.py`
- `openpilot/selfdrive/ui/sunnypilot/ui_state.py`
- `openpilot/selfdrive/ui/ui_state.py`
- model information and model settings layouts for tizi and mici;
- sidebar/home GPU status presentation; and
- UI tests covering icon color, link state, progress, readiness, and model
  identity.

The accelerator toggle controls `JetlinkEnabled`. The GPU/link presentation
reflects absent, connecting, ready, running, retrying, and unavailable states.

NRDR-specific rule: sunnypilot's settings module dynamically extends the base
`PanelType` enum. `openpilot/selfdrive/ui/layouts/main.py` must refresh its local
`PanelType` binding after importing `SettingsLayoutSP`; otherwise clicking the
NRDR home/settings card crashes with:

```text
AttributeError: type object 'PanelType' has no attribute 'NRDR'
```

Keep the regression in
`openpilot/nrdr/tests/test_settings_navigation.py` when rebasing the UI.

### Tests and developer tools

The accelerator package includes unit and seam tests for the adapter, catalog,
gadget, lending, provisioning worker, joining/fallback behavior, model state,
VM tuning, warp cache, native equivalence, selfdrived traces, and UI state.

The initial port also copied `tools/jetlink_bench.py`,
`tools/jetlink_live_bench.sh`, `tools/jetlink_replay.py`, and
`openpilot/sunnypilot/accelerators/scripts/merge_replay.sh`. Current ZoomPilot
moved the maintained comma bench/replay tools into
`jetlink_repo/scripts/comma/`. On a future port, prefer the submodule versions
and remove obsolete duplicates only after confirming no NRDR workflow still
references them.

## Recommended forward-port procedure

### 1. Record exact inputs

Start from a clean branch based on the new NRDR version. Fetch both source
branches and record their commit IDs plus the JetLink gitlink:

```bash
git fetch origin nrdr-architecture-development
git fetch zoompilot jetson-trt
git rev-parse HEAD
git rev-parse zoompilot/jetson-trt
git ls-tree zoompilot/jetson-trt jetlink_repo
```

Use a new integration branch. Do not develop directly on the rebased NRDR
release branch.

### 2. Inventory upstream changes by subsystem

Compare the old and new ZoomPilot JetLink snapshots first. This separates
JetLink evolution from changes already ported:

```bash
git log --oneline OLD_ZOOMPILOT..NEW_ZOOMPILOT -- \
  openpilot/sunnypilot/accelerators jetlink_repo

git diff --name-status OLD_ZOOMPILOT..NEW_ZOOMPILOT -- \
  openpilot/sunnypilot/accelerators \
  openpilot/sunnypilot/models \
  openpilot/selfdrive/modeld \
  openpilot/selfdrive/selfdrived \
  openpilot/selfdrive/ui \
  openpilot/system \
  jetlink_repo
```

Then compare the new NRDR base against this working branch to identify renamed
or redesigned local seams. Review diffs by subsystem instead of applying one
large patch.

### 3. Port in dependency order

Use this order so each layer has the contracts it imports:

1. Submodule, symlink, boot setup, and SCons.
2. Cereal additions and param registry entries, merged without ordinal damage.
3. Generic accelerator package and JetLink backend.
4. Model catalog, helpers, and manager integration.
5. Stock modeld and modeld_v2 joining/fallback integration.
6. Process configuration, hardware lifecycle, and shutdown.
7. Selfdrived events and transition policy.
8. UI state, model settings, link toggle, and icon/status presentation.
9. Tests, replay checks, and hardware validation.

Copy files that are wholly owned by the accelerator package when appropriate.
Manually merge shared files. The highest-risk shared files are:

```text
openpilot/cereal/custom.capnp
openpilot/common/params_keys.h
openpilot/selfdrive/modeld/modeld.py
openpilot/sunnypilot/modeld_v2/modeld.py
openpilot/sunnypilot/models/fetcher.py
openpilot/sunnypilot/models/helpers.py
openpilot/sunnypilot/models/manager.py
openpilot/selfdrive/selfdrived/selfdrived.py
openpilot/selfdrive/selfdrived/events.py
openpilot/sunnypilot/selfdrive/selfdrived/events.py
openpilot/selfdrive/ui/layouts/main.py
openpilot/system/manager/process_config.py
openpilot/system/hardware/hardwared.py
launch_chffrplus.sh
```

### 4. Audit APIs after updating the submodule

Search for stale interfaces:

```bash
rg "newer_catalogs|packed_sizes|packed_shapes" \
  openpilot/sunnypilot/accelerators openpilot/sunnypilot/models

rg "def fetch_catalogs|packed_layout|def feed_back" jetlink_repo/jetlink
rg "big_catalog\(" openpilot
```

Also inspect the current `jetlink_repo` release notes and the source branch's
submodule-bump commit. A clean import is not proof of protocol compatibility.

### 5. Initialize and verify the submodule

```bash
git submodule sync -- jetlink_repo
git submodule update --init --recursive jetlink_repo
git submodule status jetlink_repo
```

The recorded gitlink, checked-out submodule HEAD, and Python client API must
agree before testing on hardware.

## Validation

### Static and local checks

Run the tests available in the development environment, at minimum:

```bash
python -m compileall -q \
  openpilot/sunnypilot/accelerators \
  openpilot/sunnypilot/models \
  openpilot/selfdrive/modeld \
  openpilot/sunnypilot/modeld_v2

python -m unittest -v openpilot.nrdr.tests.test_settings_navigation
bash -n openpilot/sunnypilot/accelerators/setup.sh
bash -n openpilot/sunnypilot/accelerators/jetlink/setup.sh
git diff --check
```

Run the accelerator, model manager, selfdrived, and UI test suites in an
openpilot environment with all native Python dependencies. Some of these tests
need `numpy`, `pyzmq`, `pycapnp`, tinygrad, generated cereal bindings, or device
fixtures and will not import in a minimal desktop Python installation.

Before hardware deployment, confirm that NRDR-specific tests still pass. A
JetLink port is not successful if it removes or bypasses NRDR Honda behavior.

### Hardware acceptance checklist

1. Install with `git submodule update --init --recursive` completed.
2. Boot with JetLink disabled; normal small-model driving and all NRDR settings
   must still work.
3. Enable the link and verify the Jetson becomes present.
4. Select a big model and provide stable Internet while the comma is offroad.
5. Wait for download, upload, and TensorRT build to finish.
6. Verify the UI reaches ready, then running after a safe transition.
7. Confirm `modelV2.big` and the GPU/link icon change consistently.
8. Drive on the big model, then disconnect/restart the Jetson and verify clean
   fallback to the local small model without losing control output.
9. Reconnect and verify retry/rejoin behavior.
10. Open NRDR settings from the home card and exercise Honda-specific settings.
11. Reboot and confirm the previously built engine is recognized rather than
    rebuilt unnecessarily.
12. Shut down the comma and confirm the Jetson shutdown request is handled.

Initial provisioning is intentionally offroad-only. Ignition-on counts as
onroad even while the vehicle is stationary. If the drive begins during the
large LFS download, the provisioning worker is stopped and the log contains:

```text
LfsError: download interrupted
```

That message is an intentional cancellation, not proof of an invalid model URL.
For a first install, keep the comma and Jetson powered with `IsOffroad` true and
use a stable connection until the engine is ready.

## Troubleshooting map

| Symptom | First checks |
| --- | --- |
| Toggle missing | `jetlink_repo` checkout, generated `jetlink` symlink, `installed()` result, UI model layout merge |
| Jetson absent | `/dev/shm/jetlink-gadget`, USB role/cable, gadget setup, `jetlink-owner.log` |
| New big models missing | `fetch_catalogs()` API, catalog merge, submodule revision, model-manager cache |
| `server has no engine` | Normal before first provision; inspect the following download/upload/build result |
| `download interrupted` | Device transitioned onroad or worker was terminated; provision while offroad |
| Download transport error | Internet/DNS/TLS and each configured Git LFS endpoint |
| Model never joins | `JetlinkEngineReady`, `JetlinkSpec`, warp cache, joining status, safe transition window |
| Immediate modeld crash | submodule/spec API mismatch, model input layout, modeld/modeld_v2 calling convention |
| NRDR settings crash | extended `PanelType` was not rebound in `layouts/main.py` |
| Controls behavior changed | shared selfdrived/modeld files were overwritten instead of manually merged with NRDR hooks |

Useful persistent/on-device evidence includes:

```text
/data/log/jetlink-owner.log
/dev/shm/jetlink-gadget
/dev/shm/jetlink-owner-state
AcceleratorProgress
JetlinkEngineReady
JetlinkSpec
JetlinkCachedModels
Offroad_AcceleratorUnavailable
```

## Final review rule

Treat ZoomPilot as the source of truth for the evolving JetLink implementation
and NRDR as the source of truth for vehicle behavior and local architecture.
A future port is complete only when those two truths coexist: current JetLink
protocol and lifecycle behavior, with no regression or replacement of NRDR's
Honda-specific control stack.
