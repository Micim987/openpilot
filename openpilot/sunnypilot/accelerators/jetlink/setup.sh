#!/usr/bin/env bash
# Presents the comma as a USB gadget so a Jetson can enumerate it.
#
# Only when the user has turned the link on: a gadget presented by default
# turns link_configured() true and routed manager away from the user's bundle.
# The param file is read directly, by gadget.params_dir()'s rule, not through
# openpilot.common.params: this runs before build.py, and on the first boot
# after an update the params library is not built yet (the updater's git clean
# removes it). Python here also cost every boot, link on or off, 1.2 to 1.7 s.
set -u
[ -f /AGNOS ] || exit 0
BASEDIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
STATUS=/dev/shm/jetlink-gadget

# put_bool writes 1; gadget.param_bool also takes true
case "$(cat "${PARAMS_ROOT:-/data/params}/${OPENPILOT_PREFIX:-d}/JetlinkEnabled" 2>/dev/null)" in
  1|true|True) ;;
  *) exit 0 ;;
esac

REPO="$BASEDIR/jetlink_repo"
if [ ! -d "$REPO/jetlink" ]; then
  # submodule registered but never fetched; say so where the offroad alert reads
  echo "error: the jetlink package is not installed; run git submodule update --init jetlink_repo" \
    > "$STATUS" 2>/dev/null || true
  exit 0
fi

# the endpoints must exist before jetlinkd or modeld can open them, and that
# needs root. setup_gadget.sh leaves the reason in $STATUS for the offroad alert
sudo -n bash "$REPO/scripts/setup_gadget.sh" >/dev/null ||
  echo "jetlink: USB gadget setup failed" >&2
exit 0
