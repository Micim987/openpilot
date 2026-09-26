"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The gadget, and how to look at it, using nothing but the standard library.

Split out of helpers so the process that owns the gadget can be small. Holding
ep0 needs sysfs, a few params and a unix socket; `openpilot.common.swaglog`
costs 28 MB because it drags numpy, capnp and zmq in to publish a log line, and
`openpilot.common.params` imports swaglog, so a module that touches either
prices the owner out of being minimal. Measured on the comma: python plus this
plus the FunctionFS transport is 10.4 MB against 47.5 MB for the daemon that
imported the world.

Nothing here may import openpilot outside `common.hardware` (a stat) or the
jetlink package. There is a test that says so.

helpers re-exports all of it, so the heavy processes carry on calling
`helpers.udc_state()`; they also install cloudlog over `log` at import, so
their lines still reach swaglog while the owner's go to a file.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

AGNOS = os.path.isfile('/AGNOS')

# -- logging --------------------------------------------------------------
# a module-level indirection rather than an import: the owner has no swaglog
# and must not grow one, and every other process wants its lines in the drive.
log = logging.getLogger('jetlink.gadget')


def set_logger(logger) -> None:
  """Send this module's lines somewhere else; helpers points it at cloudlog."""
  global log
  log = logger


# -- params ---------------------------------------------------------------
# read straight off the filesystem. params.cc writes a value to a temp file,
# fsyncs it, renames it over the key and fsyncs the directory, so a plain read
# gets the old value or the new one and never a torn one. The path rule is
# params.cc's: PARAMS_ROOT or /data/params, plus "/" and OPENPILOT_PREFIX,
# which defaults to "d".
P_ENABLED = "JetlinkEnabled"        # user toggle; only True enables
P_READY = "JetlinkEngineReady"      # sha256 of the model the Jetson has built
P_ENDPOINT = "JetlinkEndpoint"      # optional "host:port" to use TCP instead of USB


_dirs: dict[tuple[str, str], Path] = {}


def params_dir() -> Path:
  """Where the params live, by params.cc's rule. Memoised on the two variables
  it depends on: this is on the path of every param read in the process."""
  prefix = os.environ.get('OPENPILOT_PREFIX', 'd')
  root = os.environ.get('PARAMS_ROOT', '')
  key = (root, prefix)
  found = _dirs.get(key)
  if found is None:
    # hw.h: PARAMS_ROOT, else /data/params on device. comma_home carries the
    # prefix off-device, so a bench under its own store lands where Params does
    home = root or ('/data/params' if AGNOS
                    else os.path.join(os.path.expanduser('~'),
                                      '.comma' + ('' if prefix == 'd' else prefix), 'params'))
    found = _dirs[key] = Path(home) / prefix
  return found


def raw_param(key: str) -> bytes | None:
  """A param's bytes, or None if it is unset or unreadable."""
  try:
    return (params_dir() / key).read_bytes()
  except OSError:
    return None


def param_bool(key: str) -> bool | None:
  """A param openpilot stores with put_bool. None when it is unset."""
  value = raw_param(key)
  if value is None:
    return None
  return value.strip() in (b'1', b'true', b'True')


def param_json(key: str):
  value = raw_param(key)
  if not value:
    return None
  try:
    return json.loads(value)
  except ValueError:
    return None


def enabled() -> bool:
  """Has the user switched the link on? JetlinkEnabled == True and nothing else.

  Not "absent means auto": the gadget comes up at boot with the package
  installed, so auto turned installation into enablement.
  """
  return param_bool(P_ENABLED) is True


def offroad() -> bool:
  """Is the car parked?

  The owner runs onroad too, to keep hold of the gadget, and everything else
  jetlink does belongs to a parked car: a download, a build, a warp compile.
  A missing param is manager not having written one yet, which reads as parked.
  """
  value = param_bool("IsOffroad")
  return True if value is None else value


def link_endpoint() -> tuple[str, int] | None:
  """A host:port override, for running the Jetson over ethernet during bring-up."""
  raw = raw_param(P_ENDPOINT)
  if not raw:
    return None
  host, _, port = raw.decode(errors='replace').strip().partition(':')
  return host, int(port or 5599)


# -- where the gadget lives -----------------------------------------------
# the comma is the USB gadget and the Jetson the host, decided by the kernels:
# AGNOS has CONFIG_USB_F_FS built in, L4T images are often stripped of the
# gadget modules. See jetlink/docs/transport.md
GADGET_PATH = Path("/sys/kernel/config/usb_gadget/jetlink")
FFS_MOUNT = Path("/dev/ffs-jetlink")
UDC_PATH = Path("/sys/class/udc")
# written by scripts/setup_gadget.sh, at boot or from the owner when the link is
# turned on: "ok", or "error: <reason>"
GADGET_STATUS = Path("/dev/shm/jetlink-gadget")
GADGET_SETUP_TIMEOUT = 30.0
CC_ORIENTATION = Path('/sys/class/power_supply/usb/typec_cc_orientation')
# the owner's pid while it has released the gadget on purpose so the Jetson can
# sleep. Presence comes from this, not the UDC; a marker whose writer is dead is
# a leftover from a kill
DORMANT = Path("/dev/shm/jetlink-dormant")
# hardwared's request to power the Jetson off; see backend.shutdown
SHUTDOWN_REQUEST = Path("/dev/shm/jetlink-shutdown")
# what a provisioning run leaves for the owner: whether the far end suspends
# when the gadget goes, and whether the run left anything undone. The owner
# never speaks the protocol, so it cannot learn either for itself
STATE = Path("/dev/shm/jetlink-owner-state")


def repo_root() -> Path:
  from openpilot.common.basedir import BASEDIR
  return Path(BASEDIR)


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


def bound_udc() -> str | None:
  """The device controller our gadget is attached to, if it is attached."""
  try:
    return (GADGET_PATH / "UDC").read_text().strip() or None
  except OSError:
    return None


def udc_state() -> str | None:
  """What the device controller says about the bus, or None if we are unbound.

  "configured" is a host that has us; "default" and "addressed" are one that
  reset the bus and stopped part way, which is what a Jetson that took the
  bind as a wake and did not finish waking looks like.
  """
  udc = bound_udc()
  if udc is None:
    return None
  try:
    return (UDC_PATH / udc / "state").read_text().strip() or None
  except OSError:
    return None


def host_attached() -> bool:
  """Has a host (the Jetson) enumerated and configured us?"""
  return udc_state() == "configured"


def port_has_host() -> bool:
  """Does the USB-C port controller see a host on the cable?

  The CC pin, so it is electrically true whether or not anything enumerated:
  0 is a port with nothing on it, 1 or 2 a cable with a live host. A legacy
  A-to-C cable's pull-up rides on the host's VBUS and reads the same.
  """
  try:
    return int(CC_ORIENTATION.read_text()) != 0
  except (OSError, ValueError):
    return False


# how long the UDC may sit half enumerated with a host on the cable before the
# gadget is bounced. A real enumeration is milliseconds; this only fires for a
# host that answered the bind with a bus reset and then stopped, which is what
# an unarmed hub does to a box asleep. See FfsTransport.rebind
STALLED_ENUMERATION = 20.0
STALLED_STATES = ('default', 'addressed')
HOST_POLL = 0.5


def wait_for_host(timeout: float, bounce=None, should_stop=None, report=None) -> bool:
  """Wait for the Jetson to enumerate us, bouncing a bus that stalled.

  The gadget stays bound throughout. An unbind is an unplug as the far end sees
  it, and while one boots it takes ~50 s a cycle: doing that on a timer landed
  an unplug on a box that was seconds from enumerating.

  The one case that needs an edge is a host that took the bind as a wake, reset
  the bus and stopped. The UDC then sits in default or addressed with the CC
  pin still showing a host, and only another connect moves it: that is what
  `bounce` is for, and it is spent once.
  """
  deadline = time.monotonic() + timeout
  stalled_since = None
  bounced = False
  reported = False
  while True:
    state = udc_state()
    if state == "configured":
      return True
    now = time.monotonic()
    if now >= deadline or (should_stop is not None and should_stop()):
      return False
    if report is not None and not reported:
      reported = True
      report()
    if state in STALLED_STATES and port_has_host():
      stalled_since = now if stalled_since is None else stalled_since
      if not bounced and bounce is not None and now - stalled_since > STALLED_ENUMERATION:
        bounced = True
        log.warning("jetlink: the bus has been half enumerated for %.0f s, bouncing the gadget",
                    STALLED_ENUMERATION)
        try:
          bounce()
        except Exception:
          log.exception("jetlink: could not bounce the gadget")
    else:
      stalled_since = None
    time.sleep(HOST_POLL)


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
    log.exception("jetlink: could not set up the gadget")
    return False
  log.warning("jetlink: gadget set up, the link was turned on after boot")
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


def set_dormant(on: bool) -> None:
  try:
    if on:
      DORMANT.write_text(str(os.getpid()))
    else:
      DORMANT.unlink(missing_ok=True)
  except OSError:
    log.exception("jetlink: could not update the dormant marker")


def dormant() -> bool:
  """Has a live owner released the gadget on purpose?"""
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
    log.exception("jetlink: could not write the shutdown request")
    return False


def pending_shutdown() -> str | None:
  """The reason in a shutdown request that has not been dealt with, if any.

  Read twice a second for the life of the process and almost never there, so
  the miss is a stat rather than an open that raises.
  """
  if not SHUTDOWN_REQUEST.exists():
    return None
  try:
    return str(json.loads(SHUTDOWN_REQUEST.read_text()).get('reason', ''))
  except (OSError, ValueError):
    return None


def finish_shutdown() -> None:
  try:
    SHUTDOWN_REQUEST.unlink(missing_ok=True)
  except OSError:
    log.exception("jetlink: could not remove the shutdown request")
