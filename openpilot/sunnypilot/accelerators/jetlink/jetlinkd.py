#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Owns the USB gadget, and provisions whatever large model is selected.

The comma is the USB device: the link exists only while some process holds ep0
with the UDC bound. This is that process, for as long as the link is enabled,
onroad and offroad alike. Two processes used to take turns and every change of
owner was an unplug and a replug as the Jetson saw it, a fresh libusb open and
a fresh session. modeld borrows the endpoint files instead (lending.py) and the
gadget never leaves the bus.

Provisioning (a download, an upload and a TensorRT build) takes minutes, so it
happens offroad; the result is cached on the Jetson, recorded in a param and
left loaded on the server. modeld does its own provisioning over the borrowed
link when the picked model turns out not to be built.

manager stops this on shutdown with SIGINT and SIGKILLs it 5 s later, so every
long wait polls `stop`: a FunctionFS owner killed mid-transfer leaves the
gadget in a state only a reboot clears.

Once the engine is ready and DORMANT_HOLD has passed the gadget is released so
a Jetson on an always-on supply can sleep, and only then: a server that does
not suspend keeps the link for the whole parked period. It is presented again
when there is work or when modeld borrows; the bind is the wake.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot import accelerators
from openpilot.common.params import Params
from openpilot.sunnypilot.accelerators.jetlink import helpers, lending, provision, spec_cache, warp_cache

POLL_HZ = 2.0
RETRY_BACKOFF = 30.0       # after a failed provision
RETRY_BACKOFF_MAX = 900.0  # ceiling once the failures keep coming
RECONNECT_BACKOFF = 5.0    # after the link itself failed
GADGET_SETUP_BACKOFF = 60.0  # between attempts to create a gadget boot did not

# how long after ignition-off the gadget is released once there is nothing to
# do. The server sleeps 120 s after the gadget goes; a stop inside the hold
# rejoins at once, one outside it costs the ~8 s wake
DORMANT_HOLD = 60.0
# a sleeping Jetson wakes on the bind: ~6 s to a kernel, ~1 s to enumerate
WAKE_TIMEOUT = 20.0
# how long settle() waits for its own re-enumeration before carrying on. The
# Jetson is back in about a second; the server reopens the device a poll later
SETTLE_TIMEOUT = 10.0
# after a borrower lets go. The server has just lost its client and is closing
# the gadget and reopening it, two seconds at a time; a hello inside that window
# fails, and a failed hello is read as a suspect link and costs the
# re-enumeration this whole arrangement exists to avoid
LEASE_SETTLE = 4.0

# loggerd's dirty pages pile up until the kernel reclaims them synchronously,
# right while a FunctionFS transfer allocates its buffer: gadget reads stalled
# 200-350 ms, past backend.INFERENCE_TIMEOUT, and the big model fell back.
# Capping dirty memory and holding a free-memory floor took the worst frame
# from 244 to 72 ms with no lagging frames over 20 min.
#
# System-wide, since the gadget read shares the kernel with every writer.
# Applied here so a device with the link off runs stock values, which are
# recorded in SYSCTL_PREV and put back only on disable. Never restored on
# exit: manager stops this daemon at ignition, exactly when the contention
# starts, so modeld would get stock values every drive. A reboot resets them
VM_SYSCTLS = {
  'vm.dirty_bytes': '16777216',
  'vm.dirty_background_bytes': '8388608',
  'vm.min_free_kbytes': '131072',
}
# stock AGNOS runs the dirty limits in ratio mode, so both *_bytes keys read 0,
# and the kernel silently drops a 0 written back to them. Writing the ratio key
# is what zeroes the bytes key, so the ratios are recorded alongside
VM_RATIO_KEYS = {
  'vm.dirty_bytes': 'vm.dirty_ratio',
  'vm.dirty_background_bytes': 'vm.dirty_background_ratio',
}
SYSCTL_PREV = Path('/dev/shm/jetlink-sysctl-prev')
PROC_SYS = Path('/proc/sys')


def _read_sysctls(keys) -> dict[str, str]:
  values = {}
  for key in keys:
    try:
      values[key] = (PROC_SYS / key.replace('.', '/')).read_text().strip()
    except OSError:
      cloudlog.exception(f"jetlink: could not read {key}")
  return values


def _write_sysctls(values: dict[str, str]) -> None:
  # sudo -n sysctl is the same privilege setup_gadget.sh uses; root writes /proc
  for key, value in values.items():
    try:
      if os.geteuid() == 0:
        (PROC_SYS / key.replace('.', '/')).write_text(value)
      else:
        subprocess.run(['sudo', '-n', 'sysctl', '-w', f'{key}={value}'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
      cloudlog.exception(f"jetlink: could not set {key}={value}")


def apply_vm_tuning() -> None:
  """Record the stock values once, then apply ours."""
  if not SYSCTL_PREV.exists():
    prev = _read_sysctls([*VM_SYSCTLS, *VM_RATIO_KEYS.values()])
    if prev:
      try:
        SYSCTL_PREV.write_text(json.dumps(prev))
      except OSError:
        cloudlog.exception("jetlink: could not record the previous sysctls")
  _write_sysctls(VM_SYSCTLS)


def restore_vm_tuning() -> None:
  """Put the recorded values back and drop the record."""
  try:
    prev = json.loads(SYSCTL_PREV.read_text())
  except FileNotFoundError:
    return
  except (OSError, ValueError):
    cloudlog.exception("jetlink: unreadable sysctl record, leaving the values as they are")
    prev = {}
  if isinstance(prev, dict):
    values = {}
    for key in VM_SYSCTLS:
      if key not in prev:
        continue
      ratio = VM_RATIO_KEYS.get(key)
      if str(prev[key]) == '0' and ratio in prev:
        # the kernel drops a 0 written to a *_bytes key; the ratio key is the
        # way back to ratio mode
        values[ratio] = str(prev[ratio])
      else:
        values[key] = str(prev[key])
    _write_sysctls(values)
  try:
    SYSCTL_PREV.unlink()
  except OSError:
    pass


def _timed_out(e: BaseException) -> bool:
  """Did the exchange time out with the stream still usable?

  Only LinkTimeout leaves the stream in sync; every other LinkError does not.
  """
  try:
    from jetlink.transport.base import LinkTimeout
  except ImportError:
    return False  # cannot tell, so reopen
  return isinstance(e, LinkTimeout)


class Jetlinkd:
  def __init__(self):
    self.client = None
    self.stop = False
    self.ready = False
    self.next_attempt = 0.0
    self.next_gadget_attempt = 0.0
    self.next_provision = 0.0
    self.failures = 0
    self.was_attached = False
    self.was_lent = False
    self.lease_settled = 0.0
    self.fetch_failed = False
    self.verified = False   # the server has confirmed the ready param this attach
    self.warp_built = False  # tried the comma-side warp this run
    self.warp_thread: threading.Thread | None = None
    self.started = time.monotonic()
    self.dormant = False     # released the gadget on purpose; see go_dormant
    self.vm_tuned = False    # our sysctls are in; restored only on disable
    # does the far end suspend when the gadget goes? From the server's hello,
    # and true until one says otherwise: a server too old to report it keeps
    # the release it has always had
    self.server_sleeps = True
    # modeld's lease on the endpoint files. This daemon holds ep0 and the bind
    # throughout, onroad included, so the link never leaves the bus
    self.lender = lending.Lender(self.lendable, self.bounce_gadget)

  # -- lifecycle ------------------------------------------------------------

  def request_stop(self, *_) -> None:
    self.stop = True

  def close_link(self) -> None:
    """Always go through this: a FunctionFS owner that exits without closing
    can wedge the driver until a reboot."""
    client, self.client = self.client, None
    if client is not None:
      try:
        client.close()
      except Exception:
        cloudlog.exception("jetlink: error closing the link")

  def interrupted(self) -> bool:
    """Should a long wait give up? The daemon is going away, or modeld wants
    the link.

    A build takes minutes and this daemon is no longer stopped at ignition, so
    one started while parked can still be running when the driver pulls away.
    The server's build thread carries on either way and modeld picks the engine
    up over the borrowed link, so letting go here costs nothing.
    """
    return self.stop or self.lender.lent

  def lendable(self) -> bool:
    """Is the gadget in a state modeld can take the endpoints over from?"""
    return self.client is not None and self.client.lendable

  def bounce_gadget(self) -> bool:
    """One unplug and replug, for a borrower whose write has no reader.

    Unbinding is the only thing that makes FunctionFS dequeue a write the host
    is not draining, and the unbind belongs to whoever holds ep0. It costs the
    far end a re-enumeration, so it happens on a genuine 15 s hang and nowhere
    else. See FfsTransport._abort_write.
    """
    if self.client is None:
      return False
    try:
      return bool(self.client.rebind())
    except Exception:
      cloudlog.exception("jetlink: could not bounce the gadget for the borrower")
      return False

  def settle(self) -> None:
    """Put the gadget back to bound with nothing open on it.

    ep0 and the descriptors stay here throughout; only the endpoint files go.
    See FfsTransport.release_endpoints for why that still costs the far end one
    re-enumeration, and why it is spent here, parked, right after provisioning,
    instead of at every ignition edge.

    The wait is what keeps the loop from reading its own bounce as a host
    coming and going, which would have it re-verify, provision and settle
    again for as long as the car is parked.
    """
    if self.client is None or self.lendable():
      return
    cloudlog.warning("jetlink: putting the endpoints down, keeping the gadget bound")
    if not self.client.release_endpoints():
      return   # nothing to put down: a tcp link, or one on its way out
    self.was_attached = helpers.wait_for_host(SETTLE_TIMEOUT, bounce=self.bounce_gadget,
                                              should_stop=lambda: self.stop)

  def hold_for_borrower(self) -> None:
    """Everything this daemon does once the car is moving.

    Keep the gadget on the bus, and stay off it. One sysfs read a cycle: this
    runs onroad now, and a process that wakes up to do work on modeld's core
    is a dropped frame.
    """
    if self.dormant:
      self.wake()
    if self.client is not None:
      return self.settle()      # the usual case: nothing to read, nothing to do
    if self.ensure_gadget():
      self.open_link()

  def ensure_gadget(self) -> bool:
    """Is there a gadget to present? Create it if boot did not.

    Boot only sets the gadget up with the link already on, and manager starts
    this daemon the moment the toggle flips, so a link turned on after boot
    finds nothing to open. Setting it up here is what makes the toggle act
    at once instead of at the next reboot. A device that cannot (a PC, or a
    build without the script) is left to open_link, which says why.
    """
    if helpers.link_endpoint() is not None or helpers.link_configured():
      return True
    if not helpers.can_setup_gadget():
      return True
    if time.monotonic() < self.next_gadget_attempt:
      return False
    self.next_gadget_attempt = time.monotonic() + GADGET_SETUP_BACKOFF
    return helpers.setup_gadget()

  def open_link(self) -> bool:
    """Present the gadget so a Jetson can enumerate whenever it powers on."""
    if self.client is not None:
      return True
    try:
      self.client = helpers.connect(deadline=5.0, name='jetlinkd')
      cloudlog.warning("jetlink: gadget presented, waiting for a jetson")
      return True
    except Exception:
      cloudlog.exception("jetlink: could not present the gadget")
      self.next_attempt = time.monotonic() + RECONNECT_BACKOFF
      return False

  # -- provisioning ---------------------------------------------------------

  def fetch_model(self):
    """Download the pinned large model, once.

    Minutes on a slow link, so it reports progress and stops when manager
    wants the daemon gone; it is the one call in the loop that blocks for long.
    """
    if self.fetch_failed:
      return None
    try:
      path = helpers.fetch_shipped_model(
        progress=lambda frac: accelerators.report_progress('download', frac, 'downloading the large model'),
        should_stop=self.interrupted,
      )
    except Exception:
      cloudlog.exception("jetlink: could not fetch the large model")
      accelerators.report_progress('failed', 1.0, 'could not download the large model')
      # one attempt per run; retrying a gigabyte on a loop is worse than staying small
      self.fetch_failed = True
      return None
    return path

  def build_warp(self) -> None:
    """Build a comma-side warp only if one is missing.

    scons builds it before manager starts, so this only covers a prebuilt
    image made without the target. Independent of provision(): the warp
    depends on the camera and the small model's input size, not on the
    Jetson. On a thread because the ~9 s compile cannot poll `stop` and
    manager SIGKILLs the daemon 5 s after SIGINT, which landed mid-transfer
    twice in one evening; a killed compile writes through a temporary and
    leaves nothing behind.
    """
    if self.warp_built:
      return
    self.warp_built = True
    geometry = warp_cache.device_geometry()
    if warp_cache.is_cached(*geometry):
      return
    # only past here is there a compile to report; reporting first flashed
    # "compiling the camera warp" through the UI on every start
    accelerators.report_progress('warp', 0.0, 'compiling the camera warp')

    def build() -> None:
      if warp_cache.ensure(*geometry):
        accelerators.clear_progress()

    self.warp_thread = threading.Thread(target=build, daemon=True, name='jetlink_warp')
    self.warp_thread.start()

  def provision(self) -> bool:
    """Make the Jetson ready for the selected model. Host must be attached.

    The identity comes from the catalog model's LFS pointer (the oid is the
    sha256, size the byte count), so the comma can ask without holding or
    hashing the ONNX.
    The file is only fetched when the server asks for the bytes; the Jetson
    keeps its own copy of every ONNX and never prunes it.
    """
    # imported here: the jetlink package may be absent and this module must
    # still import. Same as backend._open_link
    from jetlink.client import EngineMissing

    entry = helpers.selected_model()
    if entry is None:
      # no catalog yet; not an error
      helpers.set_engine_ready(None)
      accelerators.clear_progress()
      return False
    sha256, nbytes = provision.identity(entry)

    # the param says ready, but the Jetson's cache may have been pruned or
    # re-flashed since. Ask once per attach
    if self.verified and helpers.engine_ready_for(sha256):
      return True

    # only needed if the server turns out not to have this model; None is a
    # legitimate state here, see EngineMissing below
    model_path = helpers.shipped_model_path()

    cloudlog.warning("jetlink: provisioning %s (%d MB, sha %s)",
                     entry.get('name', sha256[:16]), nbytes >> 20, sha256[:16])
    accelerators.report_progress('connect', 0.0, 'talking to the jetson')

    hello = self.client.hello(timeout=10.0)
    Params().put('JetlinkCachedModels', hello.get('cached_models', []))
    self.note_sleep_after(hello)
    cloudlog.warning("jetlink: server %s trt %s", hello.get('device'), hello.get('trt_version'))
    try:
      spec = provision.ensure(self.client, sha256, nbytes, model_path,
                              progress=provision.report_with_eta,
                              should_stop=self.interrupted)
    except EngineMissing:
      # nothing to give. Fetch it and let the next poll try again rather than
      # holding the link through a download that takes minutes
      if model_path is None and self.fetch_model() is not None:
        return False
      raise

    cached = helpers._get('JetlinkCachedModels') or []
    Params().put('JetlinkCachedModels', sorted(set(cached) | {spec.sha256}))
    self.verified = True
    accelerators.report_progress('ready', 1.0, 'engine ready')
    cloudlog.warning("jetlink: engine ready for %s", spec.sha256[:16])
    return True

  # -- the parked car -------------------------------------------------------

  def note_sleep_after(self, hello: dict) -> None:
    """Record whether the server suspends itself when the gadget goes.

    Letting go is only worth what it costs if the Jetson sleeps when it is
    orphaned. On ignition power it does not, and releasing anyway meant a
    powered, awake box spent the whole parked period unenumerated: the icon
    read DISCONNECTED five seconds later, and every handover after that was an
    unplug the server had to recover from.
    """
    try:
      after = hello.get('sleep_after')
      self.server_sleeps = True if after is None else float(after) > 0
    except (TypeError, ValueError):
      self.server_sleeps = True
    cloudlog.warning("jetlink: the jetson %s when the gadget goes",
                     "sleeps" if self.server_sleeps else "stays up")

  def go_dormant(self) -> None:
    """Release the gadget so the Jetson can sleep. The marker goes first so
    present() never blinks. Readiness is kept; the server is asked again on
    the next attach."""
    cloudlog.warning("jetlink: nothing left to do, releasing the gadget so the jetson can sleep")
    helpers.set_dormant(True)
    self.close_link()
    self.dormant = True
    self.was_attached = False
    self.verified = False

  def wake(self) -> None:
    """Present the gadget again. If the Jetson is asleep, the bind wakes it."""
    cloudlog.warning("jetlink: presenting the gadget again")
    helpers.set_dormant(False)
    self.dormant = False

  def has_work(self) -> bool:
    """Is there a reason to wake the Jetson? Only things the link can fix
    count; the warp is local and build_warp handles it."""
    spec = spec_cache.load()
    if spec is None or not helpers.engine_ready_for(spec.sha256):
      return True
    selected = helpers.selected_model()
    return selected is not None and selected.get('oid') != spec.sha256

  def shutdown_jetson(self, reason: str) -> None:
    """hardwared is shutting the comma down and wants the Jetson off too.
    The request file is removed whatever happens: hardwared is waiting on it."""
    cloudlog.warning("jetlink: shutting the jetson down: %s", reason)
    try:
      if self.dormant:
        self.wake()
      if not self.open_link():
        raise RuntimeError("could not present the gadget")
      if not helpers.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce_gadget,
                                   should_stop=lambda: self.stop):
        raise TimeoutError(f"no jetson attached within {WAKE_TIMEOUT:.0f} s")
      resp = self.client.shutdown(reason, timeout=5.0)
      cloudlog.warning("jetlink: jetson answered the shutdown request: %s", resp)
    except Exception:
      cloudlog.exception("jetlink: could not shut the jetson down")
    finally:
      helpers.finish_shutdown()

  # -- VM tuning ------------------------------------------------------------

  def tune_vm(self) -> None:
    if not self.vm_tuned:
      apply_vm_tuning()
      self.vm_tuned = True

  def untune_vm(self) -> None:
    if self.vm_tuned:
      restore_vm_tuning()
      self.vm_tuned = False

  # -- main loop ------------------------------------------------------------

  def backoff(self) -> float:
    """Wait before provisioning again, doubling per failure.

    A powered Jetson that never answers would otherwise be retried 120 times
    an hour for as long as the car is parked, each a round of USB churn.
    """
    return min(RETRY_BACKOFF * 2 ** (self.failures - 1), RETRY_BACKOFF_MAX)

  def step(self) -> None:
    if not helpers.enabled():
      if self.client is not None or self.ready:
        cloudlog.warning("jetlink: disabled, releasing the link")
        helpers.set_engine_ready(None)
        self.ready = False
        self.close_link()
      if self.dormant:
        self.wake()
      self.untune_vm()
      return

    self.tune_vm()
    reason = helpers.pending_shutdown()
    if reason is not None and not self.lender.lent:
      # hardwared only shuts a parked car down, and a borrower has the
      # endpoints: this cannot talk over it
      self.shutdown_jetson(reason)
      return

    if not self.lender.listening and not helpers.offroad():
      # nobody can ask us for the endpoints, so ep0 in this process's hands
      # would only keep modeld out of the link for the whole drive. Give the
      # gadget up and let it own the link the way it did before there was a
      # lease: one handover, which is what this arrangement improves on, not
      # something it depends on
      self.close_link()
      return

    if self.lender.lent or not helpers.offroad():
      # onroad: hold the gadget and do nothing else. Whether or not modeld took
      # the link, a download, a build or a warp compile belongs to a parked car
      self.was_lent = self.lender.lent
      return self.hold_for_borrower()

    if self.was_lent:
      # the drive is over and the borrower has let go. Let the server finish
      # reopening before saying anything to it; see LEASE_SETTLE
      self.was_lent = False
      self.lease_settled = time.monotonic() + LEASE_SETTLE

    if self.dormant:
      if self.has_work():
        self.wake()
      else:
        return

    # before the link and the attach gate: the warp needs neither, and modeld
    # will not start the large model without it
    self.build_warp()

    if not self.ensure_gadget():
      return
    if time.monotonic() < self.next_attempt:
      return
    if not self.open_link():
      return

    attached = helpers.host_attached()
    if attached != self.was_attached:
      cloudlog.warning("jetlink: jetson %s", "attached" if attached else "gone")
      self.was_attached = attached
      if attached:
        # a host that has just arrived gets a clean slate rather than a backoff
        # earned by whatever was on the link before it
        self.failures = 0
        self.next_provision = 0.0
        self.verified = False
      else:
        # it powered down or rebooted. Readiness is about the engine on the
        # Jetson, which survives; modeld reconnects on its own
        self.ready = False
    if not attached:
      return
    # provisioning backs off on its own timer, so a long wait for an
    # unresponsive server still watches for one that reappears. The lease
    # window is its own deadline because the attach edge below clears the
    # backoff, and the borrower letting go is exactly an attach edge
    if time.monotonic() < max(self.next_provision, self.lease_settled):
      return

    try:
      self.ready = self.provision()
      self.failures = 0
      self.next_provision = 0.0
      if not self.ready:
        return
      if self.server_sleeps and time.monotonic() - self.started >= DORMANT_HOLD:
        self.go_dormant()
      else:
        # nothing more to say to the server. Put the endpoints down while the
        # car is parked, so the next drive's first borrow costs nothing
        self.settle()
    except Exception as e:
      cloudlog.exception("jetlink: provisioning failed")
      accelerators.report_progress('failed', 1.0, 'see the log')
      self.ready = False
      # only reopen when the link itself is suspect: unbinding makes the host
      # re-enumerate, and doing that for a server that is not up yet is hours
      # of USB churn
      if not _timed_out(e):
        self.close_link()
      self.failures += 1
      self.next_provision = time.monotonic() + self.backoff()

  def run(self) -> None:
    if not self.lender.start():
      cloudlog.error("jetlink: nothing can borrow the gadget from us; modeld will open it itself")
    if helpers.enabled():
      self.tune_vm()
      try:
        helpers.migrate_selection()
      except Exception:
        cloudlog.exception("jetlink: could not migrate the model selection")
    rk = Ratekeeper(POLL_HZ)
    try:
      while not self.stop:
        rk.keep_time()
        try:
          self.step()
        except Exception:
          # nothing may escape: restarting in a loop is worse than sitting out a cycle
          cloudlog.exception("jetlink: unhandled error")
          self.close_link()
          self.next_attempt = time.monotonic() + RECONNECT_BACKOFF
    finally:
      # the sysctls stay: a stop here is where a drive begins
      self.lender.stop()
      self.close_link()
    helpers.set_dormant(False)
    if self.warp_thread is not None and self.warp_thread.is_alive():
      cloudlog.warning("jetlink: stopped with the warp still compiling; it will rebuild next time")
    cloudlog.warning("jetlink: stopped")


def main() -> None:
  d = Jetlinkd()
  # manager stops this at the onroad transition; closing the link properly is
  # what keeps the driver healthy for modeld
  signal.signal(signal.SIGTERM, d.request_stop)
  signal.signal(signal.SIGINT, d.request_stop)
  d.run()


if __name__ == "__main__":
  main()
