#!/usr/bin/env bash
# Presents the comma as a USB gadget so a Jetson can enumerate it, and links
# the standalone jetlink package onto PYTHONPATH.
#
# The package may live outside the repo until it is a submodule: the updater's
# reset --hard + clean deletes untracked files inside the repo, which would
# take it with it.
set -u
BASEDIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
STATUS=/dev/shm/jetlink-gadget

REPO=""
for candidate in "$BASEDIR/jetlink_repo" /data/jetlink_repo; do
  [ -d "$candidate" ] && { REPO="$candidate"; break; }
done

if [ -z "$REPO" ]; then
  # Most likely a fresh switch to this branch with the submodule never fetched.
  # Say so where the offroad alert can find it rather than leaving the feature
  # mysteriously absent.
  echo "error: the jetlink package is not installed; expected jetlink_repo/ or /data/jetlink_repo" \
    > "$STATUS" 2>/dev/null || true
  exit 0
fi

ln -sfn "$REPO/jetlink" "$BASEDIR/jetlink"
[ -f /AGNOS ] || exit 0

# Do not silently pair a new fork with an old manually copied client package.
# Validation is startup-only, before modeld or the gadget owner runs.
LOCK="$BASEDIR/openpilot/sunnypilot/accelerators/jetlink/release.json"
if ! python3 "$REPO/scripts/verify_release.py" "$REPO" "$LOCK" >/dev/null; then
  echo "error: jetlink package does not match this openpilot release; reinstall the pinned package" > "$STATUS"
  exit 0
fi

# The endpoints have to exist before jetlinkd or modeld can open them, and
# configuring a gadget needs root while the launcher runs as comma. Do not
# silence a failure: setup_gadget.sh leaves the reason in $STATUS, which is what
# the offroad alert reads.
sudo -n bash "$REPO/scripts/setup_gadget.sh" >/dev/null ||
  echo "jetlink: USB gadget setup failed" >&2

# Keep a recording-writeback storm from stalling the gadget read path.
# loggerd/encoderd write video continuously; on a memory-tight comma (stock
# min_free_kbytes ~7 MB, ~40 MB free) a segment's dirty pages pile up until the
# kernel has to reclaim them synchronously - write them back before it can
# evict them - exactly while a FunctionFS transfer is allocating its buffer.
# Measured on the 2026-09-06 bench: nr_dirty to 108 MB, direct reclaim, and the
# gadget read stalled 200-350 ms, past backend.INFERENCE_TIMEOUT, so the big
# model fell back and rejoined (91 lagging frames, three losses in 15 min).
# Capping dirty memory (so reclaim finds clean, evictable pages) and holding a
# real free-memory floor (so allocations do not reclaim at all) removed it:
# dirty peak 7 MB, worst frame 244 -> 72 ms, zero lagging frames over 20 min;
# under a 500 MB memory hog plus CPU contention, 128 MB/16 MB held it too.
# System-wide on purpose - the gadget read shares the kernel with every writer.
for kv in vm.dirty_bytes=16777216 vm.dirty_background_bytes=8388608 vm.min_free_kbytes=131072; do
  sudo -n sysctl -w "$kv" >/dev/null 2>&1 || echo "jetlink: could not set $kv" >&2
done
exit 0
