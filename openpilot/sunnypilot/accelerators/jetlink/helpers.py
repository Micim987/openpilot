"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Where the model is, whether the Jetson is attached, and how far along it is.

The list of large models is sunnypilot's big-model catalog, which the model
manager already fetches and caches on every device. Its bundles are tinygrad
pkls for a GPU the Jetson does not have, but each one names the comma commit
it was compiled from, and that commit's ONNX in comma's LFS is what the Jetson
runs. The catalog says what exists; the pointer at the commit says which bytes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from openpilot.common.hardware import AGNOS
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.models.model_name import DEFAULT_BIG_MODEL_REF

# none of these are CLEAR_ON_MANAGER_START: readiness must survive a reboot or
# every ignition rebuilds a 160 s engine
P_ENABLED = "JetlinkEnabled"        # user toggle; only True enables
P_READY = "JetlinkEngineReady"      # sha256 of the model the Jetson has built
P_ENDPOINT = "JetlinkEndpoint"      # optional "host:port" to use TCP instead of USB

UNCHUNKED_SUFFIX = ".jetlink-unchunked"


def _get(key: str, default=None):
  """Read a param, tolerating a params library that predates the key.

  Called from hardwared and the UI's param thread, so UnknownKeyName here
  would take down a process that has nothing to do with jetlink.
  """
  try:
    return Params().get(key)
  except Exception:
    return default


def link_endpoint() -> tuple[str, int] | None:
  """A host:port override, for running the Jetson over ethernet during bring-up."""
  raw = _get(P_ENDPOINT)
  if not raw:
    return None
  host, _, port = raw.strip().partition(':')
  return host, int(port or 5599)


# the comma is the USB gadget and the Jetson the host, decided by the kernels:
# AGNOS has CONFIG_USB_F_FS built in, L4T images are often stripped of the
# gadget modules. See jetlink/docs/transport.md
GADGET_PATH = Path("/sys/kernel/config/usb_gadget/jetlink")
FFS_MOUNT = Path("/dev/ffs-jetlink")
UDC_PATH = Path("/sys/class/udc")
# written by scripts/setup_gadget.sh, at boot or from jetlinkd when the link is
# turned on: "ok", or "error: <reason>"
GADGET_STATUS = Path("/dev/shm/jetlink-gadget")
GADGET_SETUP_TIMEOUT = 30.0
CC_ORIENTATION = Path('/sys/class/power_supply/usb/typec_cc_orientation')


def gadget_error() -> str | None:
  """Why the USB gadget is unavailable, if it is.

  The gadget is set up at boot by root from launch_chffrplus.sh, nowhere a
  user would look. A missing file is not an error: the setup never ran.
  """
  try:
    reason = GADGET_STATUS.read_text().strip()
  except OSError:
    return None
  if not reason or reason == 'ok':
    return None
  return reason.removeprefix('error:').strip() or None


def gadget_bound() -> bool:
  """Has our gadget been attached to a device controller?"""
  try:
    return bool((GADGET_PATH / "UDC").read_text().strip())
  except OSError:
    return False


def host_attached() -> bool:
  """Has a host (the Jetson) enumerated and configured us?"""
  try:
    udc = (GADGET_PATH / "UDC").read_text().strip()
  except OSError:
    return False
  if not udc:
    return False
  try:
    return (UDC_PATH / udc / "state").read_text().strip() == "configured"
  except OSError:
    return False


def package_installed() -> bool:
  """Is the jetlink submodule checked out? A stat, not an import: the UI asks at 5 Hz."""
  try:
    return (repo_root() / 'jetlink_repo' / 'jetlink' / '__init__.py').is_file()
  except OSError:
    return False


def _gadget_script() -> Path:
  return repo_root() / 'jetlink_repo' / 'scripts' / 'setup_gadget.sh'


def can_setup_gadget() -> bool:
  """Only AGNOS has the gadget stack, and only the submodule has the script."""
  return AGNOS and _gadget_script().is_file()


def setup_gadget() -> bool:
  """Create the gadget the way boot does. The link was off at boot and is on now.

  The script records "ok" or the reason in GADGET_STATUS itself, so a failure
  here reaches the offroad alert the same way a failure at boot does.
  """
  try:
    subprocess.run(['sudo', '-n', 'bash', str(_gadget_script())], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=GADGET_SETUP_TIMEOUT)
  except Exception:
    cloudlog.exception("jetlink: could not set up the gadget")
    return False
  cloudlog.warning("jetlink: gadget set up, the link was turned on after boot")
  return link_configured()


def link_configured() -> bool:
  """Can we even attempt a link? The gadget exists, or TCP is configured.

  Not host_attached(): the UDC only binds when something opens ep0, and nothing
  opens ep0 unless the link looks usable. Waiting for a host deadlocks.
  """
  if gadget_error() is not None:
    return False
  if link_endpoint() is not None:
    return True
  try:
    return (FFS_MOUNT / "ep0").exists()
  except OSError:
    # a root-only mount raises PermissionError from stat; unusable either way
    return False


# jetlinkd's pid while it has released the gadget on purpose so the Jetson can
# sleep (Jetlinkd.go_dormant). Presence comes from this, not the UDC; a marker
# whose writer is dead is a leftover from a kill
DORMANT = Path("/dev/shm/jetlink-dormant")
# hardwared's request to power the Jetson off; see backend.shutdown
SHUTDOWN_REQUEST = Path("/dev/shm/jetlink-shutdown")


def set_dormant(on: bool) -> None:
  try:
    if on:
      DORMANT.write_text(str(os.getpid()))
    else:
      DORMANT.unlink(missing_ok=True)
  except OSError:
    cloudlog.exception("jetlink: could not update the dormant marker")


def dormant() -> bool:
  """Has a live jetlinkd released the gadget on purpose?"""
  try:
    pid = int(DORMANT.read_text())
  except (OSError, ValueError):
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    pass  # alive, just not ours to signal
  return True


def request_shutdown(reason: str) -> bool:
  try:
    SHUTDOWN_REQUEST.write_text(json.dumps({'reason': reason}))
    return True
  except OSError:
    cloudlog.exception("jetlink: could not write the shutdown request")
    return False


def pending_shutdown() -> str | None:
  """The reason in a shutdown request that has not been dealt with, if any."""
  try:
    return str(json.loads(SHUTDOWN_REQUEST.read_text()).get('reason', ''))
  except (OSError, ValueError):
    return None


def finish_shutdown() -> None:
  try:
    SHUTDOWN_REQUEST.unlink(missing_ok=True)
  except OSError:
    cloudlog.exception("jetlink: could not remove the shutdown request")


def await_shutdown(timeout: float) -> bool:
  """Wait for jetlinkd to take the request. False if nobody did in time."""
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not SHUTDOWN_REQUEST.exists():
      return True
    time.sleep(0.25)
  finish_shutdown()
  return False


# a USB3 link recovery passes through "addressed" for a moment, and presence
# read at 2 Hz should not blink for it
PRESENCE_HOLD = 5.0
_last_configured = 0.0


def gadget_present() -> bool:
  """Is a Jetson actually on the other end right now?

  True once something holds the gadget open and a host has configured us,
  held for PRESENCE_HOLD after that stops.
  """
  global _last_configured
  if link_endpoint() is not None:
    return True
  if dormant():
    # no enumeration during suspend; the CC line still tells a sleeping host from an unplugged one
    try:
      return int(CC_ORIENTATION.read_text()) != 0
    except (OSError, ValueError):
      return False
  now = time.monotonic()
  if host_attached():
    _last_configured = now
    return True
  return now - _last_configured < PRESENCE_HOLD


def connect(deadline: float | None = None):
  """Open the link. USB unless an endpoint override is set.

  `deadline` is per frame and defaults to FRAME_TIMEOUT: modeld blocks on a
  frame the way it blocks on a chestnut.
  """
  from jetlink.client import FRAME_TIMEOUT, JetlinkClient
  deadline = FRAME_TIMEOUT if deadline is None else deadline
  endpoint = link_endpoint()
  if endpoint is not None:
    host, port = endpoint
    cloudlog.warning("jetlink: connecting over tcp to %s:%d", host, port)
    return JetlinkClient.open_tcp(host, port, deadline=deadline)
  # the comma is the gadget and the Jetson the host; see gadget_present()
  return JetlinkClient.open_ffs(str(FFS_MOUNT), gadget=str(GADGET_PATH), deadline=deadline)


def enabled() -> bool:
  """Has the user switched the link on? JetlinkEnabled == True and nothing else.

  Not "absent means auto": the gadget comes up at boot with the package
  installed, so auto turned installation into enablement.
  """
  return bool(_get(P_ENABLED))


def gadget_alert() -> str | None:
  """The gadget failure worth an alert: only for someone who asked for the link."""
  return gadget_error() if enabled() else None


# -- the model ------------------------------------------------------------

def active_bundle():
  from openpilot.sunnypilot.models.helpers import get_selected_bundle
  return get_selected_bundle(Params(), "chestnut")


def _artifact_names(bundle) -> list[str]:
  out = []
  for model in getattr(bundle, 'models', []) or []:
    artifact = getattr(model, 'artifact', None)
    name = getattr(artifact, 'fileName', None) if artifact else None
    if name and name.endswith('.onnx'):
      out.append(name)
  return out


def active_model_path() -> Path | None:
  """Path to the selected large model's ONNX, materialising chunks if needed.

  No bundle selected is the normal case: no model-manager bundle ships an
  ONNX, so a Jetson runs the catalog model's ONNX fetched from LFS. None means
  no large model here yet and the caller stays on the small model.
  """
  bundle = active_bundle()
  root = Path(Paths.model_root())
  for name in _artifact_names(bundle) if bundle is not None else []:
    plain = root / name
    if plain.is_file() and plain.stat().st_size > 1_000_000:
      return plain
    # chunked download: reassemble once, next to the chunks
    manifest = root / f'{name}.chunkmanifest'
    if manifest.is_file():
      return _materialise(root / name)
  return shipped_model_path()


P_MODEL = "JetlinkModel"             # the chosen catalog bundle's ref, a comma commit
P_POINTERS = "JetlinkModelPointers"  # ref -> {oid, size}; a commit's tree never changes
# the model manager's copy of sunnypilot's big-model catalog; the key is
# models.fetcher.ModelFetcher.MODEL_SOURCES['chestnut']'s
CATALOG_PARAM = "ModelManager_ModelsCache_Chestnut"

# comma overwrites this one file, so the commit is the only name a big model's
# ONNX has. GitHub's raw host serves the LFS pointer for any commit it holds,
# merged or not, and the pointer is the oid and size the Jetson is asked for
POINTER_URL = 'https://raw.githubusercontent.com/commaai/openpilot/{ref}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx'
POINTER_TIMEOUT = 10.0
_REF = re.compile(r'[0-9a-f]{40}')
# the index is two JSON params, and the UI names the active model every frame;
# a picker opened a moment after a catalog refresh can wait this long
INDEX_TTL = 2.0
_index_cache: tuple[float, list[dict]] | None = None


def repo_root() -> Path:
  return Path(__file__).resolve().parents[4]


def catalog() -> list[dict]:
  """sunnypilot's big-model bundles as {name, ref, folder}, newest first, as the
  model manager's own picker lists them.

  From the cached JSON rather than the parsed bundles: parsing builds capnp
  objects and writes chunk manifests, for three fields. Read from the UI's
  param thread, so nothing escapes.
  """
  try:
    from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION
    bundles = (Params().get(CATALOG_PARAM) or {}).get('bundles', [])
    found = [b for b in bundles if _REF.fullmatch(str(b.get('ref')))
             and int(b.get('minimum_selector_version', 0)) == REQUIRED_JSON_VERSION]
  except Exception:
    cloudlog.exception("jetlink: could not read the big-model catalog")
    return []
  found.sort(key=lambda b: int(b.get('index', 0)), reverse=True)
  return [{'name': str(b.get('display_name') or b['ref'][:10]), 'ref': b['ref'],
           'folder': str((b.get('overrides') or {}).get('folder', ''))} for b in found]


def pointers() -> dict[str, dict]:
  value = _get(P_POINTERS)
  return value if isinstance(value, dict) else {}


def fetch_pointer(ref: str) -> tuple[str, int]:
  """The oid and size of the ONNX at a comma commit."""
  from openpilot.sunnypilot.accelerators.jetlink import lfs
  with urllib.request.urlopen(POINTER_URL.format(ref=ref), timeout=POINTER_TIMEOUT) as response:
    text = response.read(4096).decode()
  parsed = lfs.parse_pointer_text(text)
  if parsed is None:
    raise ValueError(f"{ref[:10]} did not serve an lfs pointer")
  return parsed


def resolve_pointer(ref: str) -> tuple[str, int]:
  """The oid and size behind a catalog model, fetched the first time and kept for good."""
  global _index_cache
  known = pointers()
  if ref in known:
    return known[ref]['oid'], int(known[ref]['size'])
  oid, size = fetch_pointer(ref)
  known[ref] = {'oid': oid, 'size': size}
  # blocking: the next lookup reads this back, and a put still in flight would be lost under it
  Params().put(P_POINTERS, known, block=True)
  _index_cache = None
  cloudlog.warning("jetlink: %s is %s, %d MB", ref[:10], oid[:16], size >> 20)
  return oid, size


def model_index() -> list[dict]:
  """Every catalog model, {name, ref, folder, oid, size}. oid and size are None
  until the model has been selected and resolved. No network."""
  global _index_cache
  now = time.monotonic()
  if _index_cache is not None and now - _index_cache[0] < INDEX_TTL:
    return _index_cache[1]
  known = pointers()
  out = []
  for b in catalog():
    p = known.get(b['ref']) or {}
    out.append({**b, 'oid': p.get('oid'), 'size': int(p['size']) if p.get('size') else None})
  _index_cache = (now, out)
  return out


def selected_model(models: list[dict] | None = None) -> dict | None:
  """The entry the user picked, or the fork's default big model, or the newest.

  A ref the catalog dropped falls back the same way: leaving the device with
  no model at all would be worse than quietly using the default.
  """
  models = model_index() if models is None else models
  if not models:
    return None
  wanted = _get(P_MODEL)
  if wanted:
    for m in models:
      if m['ref'] == wanted:
        return m
    cloudlog.warning("jetlink: no catalog model for %r, using the default", wanted)
  return next((m for m in models if m['ref'] == DEFAULT_BIG_MODEL_REF), models[0])


def migrate_selection() -> None:
  """A JetlinkModel written before the catalog holds an index name.

  Rewrite it as the ref of the model that is provisioned, so the engine on the
  Jetson is kept, or clear it so the default applies. jetlinkd, once at start:
  it costs a pointer fetch per catalog model until the match, and is left for
  the next run if the network is not there.
  """
  wanted = _get(P_MODEL)
  if not wanted or _REF.fullmatch(wanted):
    return
  ready = _get(P_READY)
  ref = None
  for m in (catalog() if ready else []):
    try:
      oid, _ = resolve_pointer(m['ref'])
    except Exception as e:
      cloudlog.warning("jetlink: cannot migrate the selection %r yet: %s", wanted, e)
      return
    if oid == ready:
      ref = m['ref']
      break
  if ref:
    Params().put(P_MODEL, ref, block=True)
    cloudlog.warning("jetlink: selection %r is now %s, the model that is provisioned", wanted, ref[:10])
  else:
    Params().remove(P_MODEL)
    cloudlog.warning("jetlink: selection %r is not a catalog model, using the default", wanted)


def model_dir() -> Path:
  """Ours, under the model manager's root: its cache clear removes every file
  it does not recognise and leaves directories alone."""
  return Path(Paths.model_root()) / 'jetlink'


def model_file_name(model: dict) -> str:
  """One file per model, so switching back does not re-download."""
  return f"{model['oid'][:16]}.onnx"


def shipped_model_path() -> Path | None:
  """The chosen large model, if it has been fetched.

  Keyed on the oid, not the in-tree pointer, which moves with upstream syncs.
  Size is the cheap check that the file is the one we mean.
  """
  model = selected_model()
  if model is None or not model['oid']:
    return None
  path = model_dir() / model_file_name(model)
  if path.is_file() and path.stat().st_size == model['size']:
    return path
  return None


def _materialise(path: Path) -> Path | None:
  from openpilot.common.file_chunker import open_file_chunked
  out = path.with_name(path.name + UNCHUNKED_SUFFIX)
  if out.is_file() and out.stat().st_size > 1_000_000:
    return out
  free = shutil.disk_usage(path.parent).free
  try:
    with open_file_chunked(str(path)) as src, open(out, 'wb') as dst:
      shutil.copyfileobj(src, dst, length=4 << 20)
  except Exception:
    cloudlog.exception("jetlink: could not reassemble %s (%d MB free)", path.name, free >> 20)
    out.unlink(missing_ok=True)
    return None
  return out


def cleanup_unchunked(keep: Path | None = None) -> None:
  root = Path(Paths.model_root())
  for p in root.glob(f'*{UNCHUNKED_SUFFIX}'):
    if keep is None or p != keep:
      p.unlink(missing_ok=True)


def fetch_shipped_model(progress=None, should_stop=None) -> Path | None:
  """Download the chosen large model if it is not here yet. None when nothing is chosen."""
  from openpilot.sunnypilot.accelerators.jetlink import lfs
  model = selected_model()
  if model is None or not model['oid']:
    return None
  dest = model_dir() / model_file_name(model)
  return lfs.fetch_oid(model['oid'], model['size'], dest, repo_root(),
                       progress=progress, should_stop=should_stop)


# -- readiness ------------------------------------------------------------

def engine_ready_for(sha256: str | None) -> bool:
  if not sha256:
    return False
  return (_get(P_READY) or '') == sha256


def set_engine_ready(sha256: str | None) -> None:
  params = Params()
  if sha256:
    params.put(P_READY, sha256)
  else:
    params.remove(P_READY)
