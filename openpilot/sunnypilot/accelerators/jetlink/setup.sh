#!/usr/bin/env bash
# Presents the comma as a USB gadget so a Jetson can enumerate it.
#
# Only when the user has turned the link on. Installing the package must not
# enable the feature: a gadget presented by default turns link_configured()
# true on every device with the package, and that used to route manager away
# from the user's small-model bundle without anyone asking for it. The param
# is read through openpilot.common.params so the key's type and prefix are
# the ones every other reader uses; a params library that is not built yet
# (a fresh checkout before build.py), or one without the key, reads as off.
set -u
[ -f /AGNOS ] || exit 0
BASEDIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
STATUS=/dev/shm/jetlink-gadget

enabled=$(PYTHONPATH="$BASEDIR" python3 - <<'PY' 2>/dev/null || echo 0
try:
  from openpilot.common.params import Params
  print(1 if Params().get_bool("JetlinkEnabled") else 0)
except Exception:
  print(0)
PY
)
[ "$enabled" = "1" ] || exit 0

REPO="$BASEDIR/jetlink_repo"
if [ ! -d "$REPO/jetlink" ]; then
  # A submodule registered but never fetched. Say so where the offroad alert
  # can find it rather than leaving the feature mysteriously absent.
  echo "error: the jetlink package is not installed; run git submodule update --init jetlink_repo" \
    > "$STATUS" 2>/dev/null || true
  exit 0
fi

# The endpoints have to exist before jetlinkd or modeld can open them, and
# configuring a gadget needs root while the launcher runs as comma. Do not
# silence a failure: setup_gadget.sh leaves the reason in $STATUS, which is what
# the offroad alert reads.
sudo -n bash "$REPO/scripts/setup_gadget.sh" >/dev/null ||
  echo "jetlink: USB gadget setup failed" >&2
exit 0
