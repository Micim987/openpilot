#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Holds the USB gadget, and nothing else.

The comma is the USB device: the link exists only while some process holds ep0
with the UDC bound. This is that process, for as long as the link is enabled,
onroad and offroad alike. Whoever wants to move bytes borrows the endpoint
files over a unix socket (lending.py) and the gadget never leaves the bus.

It is deliberately small. Everything heavy jetlink does is episodic, so none of
it lives here: a download, an upload, a TensorRT build and a warp compile all
belong to jetlinkd, which this spawns when there is something to do and which
exits when there is not. That keeps a parked car and a drive alike at one
resident jetlink process of about 13 MB rather than 47.5 MB, and it is why
nothing in this module may import swaglog, Params, numpy, capnp or zmq; see
gadget.py and tests/test_gadget.py.

manager stops this on shutdown with SIGINT and SIGKILLs it 5 s later, so every
long wait polls `stop`: a FunctionFS owner killed mid-transfer leaves the
gadget in a state only a reboot clears.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget, lending, vmtune

POLL = 0.5
# how long after ignition-off the gadget is released once there is nothing to
# do. The server sleeps 120 s after the gadget goes; a stop inside the hold
# rejoins at once, one outside it costs the ~8 s wake
DORMANT_HOLD = 60.0
# how long settle() waits for its own re-enumeration before carrying on
SETTLE_TIMEOUT = 10.0
# after a borrower lets go. The server has just lost its client and is closing
# the gadget and reopening it, two seconds at a time
LEASE_SETTLE = 4.0
RECONNECT_BACKOFF = 5.0
GADGET_SETUP_BACKOFF = 60.0
# between runs of the worker that found nothing to do. It costs a couple of
# seconds of imports, so it is spawned on a change and not on a timer
WORKER_BACKOFF = 300.0
WORKER_GRACE = 10.0

LOG = Path('/dev/shm/jetlink-owner.log')
# what the worker leaves behind for us: whether the far end suspends when the
# gadget goes, and whether it had anything left to do
STATE = Path('/dev/shm/jetlink-owner-state')
# params whose change is a reason to look again: the pick, and what is built
WATCHED = ('ModelManager_ActiveBundleChestnut', 'JetlinkEngineReady', 'JetlinkSpec')
WORKER = 'openpilot.sunnypilot.accelerators.jetlink.jetlinkd'


def _log_to_file() -> logging.Logger:
  """swaglog costs 28 MB, so the owner keeps its own. The worker's lines go to
  the drive as they always did; these are for a bench session."""
  log = logging.getLogger('jetlink.owner')
  log.setLevel(logging.INFO)
  try:
    handler: logging.Handler = logging.FileHandler(LOG)
  except OSError:
    handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
  log.addHandler(handler)
  log.addHandler(logging.StreamHandler())
  return log


class Owner:
  def __init__(self):
    self.transport = None
    self.stop = False
    self.dormant = False
    self.vm_tuned = False
    self.started = time.monotonic()
    self.next_attempt = 0.0
    self.next_gadget_attempt = 0.0
    self.next_worker = 0.0
    self.lease_settled = 0.0
    self.was_lent = False
    self.worker: subprocess.Popen | None = None
    self.seen: dict[str, int] = {}      # watched param -> mtime when last looked
    self.had_host = False
    self.lender = lending.Lender(self.lendable, self.bounce_gadget)

  # -- the gadget -----------------------------------------------------------

  def close_link(self) -> None:
    """Always go through this: a FunctionFS owner that exits without closing
    can wedge the driver until a reboot."""
    transport, self.transport = self.transport, None
    if transport is not None:
      try:
        transport.close()
      except Exception:
        gadget.log.exception("jetlink: error closing the link")

  def lendable(self) -> bool:
    """Is the gadget in a state a borrower can take the endpoints over from?"""
    return self.transport is not None and self.transport.lendable

  def bounce_gadget(self) -> bool:
    """One unplug and replug, for a borrower whose write has no reader.

    Unbinding is the only thing that makes FunctionFS dequeue a write the host
    is not draining, and the unbind belongs to whoever holds ep0. See
    FfsTransport._abort_write.
    """
    if self.transport is None:
      return False
    try:
      return bool(self.transport.rebind())
    except Exception:
      gadget.log.exception("jetlink: could not bounce the gadget for the borrower")
      return False

  def open_link(self) -> bool:
    """Present the gadget so a Jetson can enumerate whenever it powers on.

    The transport, not a client: this end never speaks the protocol, and
    jetlink.client would bring numpy with it.
    """
    if self.transport is not None:
      return True
    if gadget.link_endpoint() is not None:
      return False   # the Jetson is on ethernet; there is no gadget to own
    try:
      from jetlink.transport.ffs import FfsTransport
      self.transport = FfsTransport(str(gadget.FFS_MOUNT), gadget=str(gadget.GADGET_PATH))
      gadget.log.warning("jetlink: gadget presented, waiting for a jetson")
      return True
    except Exception:
      gadget.log.exception("jetlink: could not present the gadget")
      self.next_attempt = time.monotonic() + RECONNECT_BACKOFF
      return False

  def ensure_gadget(self) -> bool:
    """Is there a gadget to present? Create it if boot did not.

    Boot only sets the gadget up with the link already on, so a link turned on
    afterwards finds nothing to open. Setting it up here is what makes the
    toggle act at once instead of at the next reboot.
    """
    if gadget.link_endpoint() is not None or gadget.link_configured():
      return True
    if not gadget.can_setup_gadget():
      return True
    if time.monotonic() < self.next_gadget_attempt:
      return False
    self.next_gadget_attempt = time.monotonic() + GADGET_SETUP_BACKOFF
    return gadget.setup_gadget()

  def settle(self) -> None:
    """Put the gadget back to bound with nothing open on it.

    ep0 and the descriptors stay here throughout; only the endpoint files go.
    See FfsTransport.release_endpoints.
    """
    if self.transport is None or self.lendable():
      return
    gadget.log.warning("jetlink: putting the endpoints down, keeping the gadget bound")
    if not self.transport.release_endpoints():
      return
    gadget.wait_for_host(SETTLE_TIMEOUT, bounce=self.bounce_gadget,
                         should_stop=lambda: self.stop)

  def hold(self) -> None:
    """Everything this process does once the car is moving, or once somebody
    has the endpoints.

    Keep the gadget on the bus and stay off it. One sysfs read a cycle: a
    process that wakes up to do work on modeld's core is a dropped frame.
    """
    if self.dormant:
      self.wake()
    if self.transport is not None:
      return self.settle()
    if self.ensure_gadget():
      self.open_link()

  # -- the parked car -------------------------------------------------------

  def server_sleeps(self) -> bool:
    """Does the far end suspend when the gadget goes? The worker learns it from
    the server's hello and leaves it here. True until one says otherwise: a
    server too old to report it keeps the release it has always had."""
    return self.state().get('sleep_after', 1.0) > 0

  def state(self) -> dict:
    try:
      value = json.loads(STATE.read_text())
    except (OSError, ValueError):
      return {}
    return value if isinstance(value, dict) else {}

  def go_dormant(self) -> None:
    """Release the gadget so the Jetson can sleep. The marker goes first so
    present() never blinks."""
    gadget.log.warning("jetlink: nothing left to do, releasing the gadget so the jetson can sleep")
    gadget.set_dormant(True)
    self.close_link()
    self.dormant = True
    self.had_host = False

  def wake(self) -> None:
    """Present the gadget again. If the Jetson is asleep, the bind wakes it."""
    gadget.log.warning("jetlink: presenting the gadget again")
    gadget.set_dormant(False)
    self.dormant = False

  # -- the worker -----------------------------------------------------------

  def marks(self) -> dict[str, int]:
    """When each watched param last changed. A stat, not a read: the owner does
    not parse the catalog or the spec, it only notices they moved."""
    out = {}
    for key in WATCHED:
      try:
        out[key] = (gadget.params_dir() / key).stat().st_mtime_ns
      except OSError:
        out[key] = 0
    return out

  def worker_running(self) -> bool:
    if self.worker is None:
      return False
    if self.worker.poll() is None:
      return True
    gadget.log.warning("jetlink: the provisioning run finished (%s)", self.worker.returncode)
    self.worker = None
    return False

  def wanted(self) -> str | None:
    """Why the worker should run, or None. Everything that decides whether
    there is work needs the catalog, the spec and the Jetson, so the worker
    decides; this only notices the things that could have changed the answer."""
    if gadget.pending_shutdown() is not None:
      return 'the jetson has to be shut down'
    marks = self.marks()
    if not self.seen:
      return 'nothing has been checked since boot'
    changed = [k for k, v in marks.items() if self.seen.get(k) != v]
    if changed:
      return f"{', '.join(changed)} changed"
    if gadget.host_attached() and not self.had_host:
      return 'a jetson turned up'
    if time.monotonic() >= self.next_worker and self.state().get('unfinished'):
      return 'the last run left something to do'
    return None

  def spawn_worker(self, why: str) -> None:
    gadget.log.warning("jetlink: starting a provisioning run: %s", why)
    self.seen = self.marks()
    self.had_host = gadget.host_attached()
    self.next_worker = time.monotonic() + WORKER_BACKOFF
    try:
      self.worker = subprocess.Popen([sys.executable, '-m', WORKER],
                                     cwd=str(gadget.repo_root()),
                                     env={**os.environ, 'PYTHONPATH': str(gadget.repo_root())})
    except Exception:
      gadget.log.exception("jetlink: could not start the provisioning run")
      self.worker = None

  def stop_worker(self) -> None:
    if self.worker is None:
      return
    self.worker.terminate()
    try:
      self.worker.wait(WORKER_GRACE)
    except subprocess.TimeoutExpired:
      self.worker.kill()
    self.worker = None

  # -- the loop -------------------------------------------------------------

  def request_stop(self, *_) -> None:
    self.stop = True

  def step(self) -> None:
    if not gadget.enabled():
      if self.transport is not None:
        gadget.log.warning("jetlink: disabled, releasing the link")
        self.close_link()
      self.stop_worker()
      if self.dormant:
        self.wake()
      if self.vm_tuned:
        vmtune.restore_vm_tuning()
        self.vm_tuned = False
      return

    if not self.vm_tuned:
      vmtune.apply_vm_tuning()
      self.vm_tuned = True

    if not gadget.offroad():
      # the drive has started and the endpoints belong to modeld. A run of ours
      # holding the lease would keep it out for the whole drive; the server's
      # build carries on and modeld picks the engine up over its own link
      self.stop_worker()

    if self.lender.lent or not gadget.offroad():
      self.was_lent = self.lender.lent
      return self.hold()

    if self.was_lent:
      # the borrower has just let go and the server is still reopening the
      # gadget it lost; a hello inside that window fails and costs a
      # re-enumeration. Its own deadline, because a host arriving clears the
      # worker's backoff and a borrower letting go looks like one arriving
      self.was_lent = False
      self.lease_settled = time.monotonic() + LEASE_SETTLE

    if self.worker_running():
      return self.hold()

    if not self.ensure_gadget():
      return
    if time.monotonic() < self.next_attempt:
      return

    why = self.wanted() if time.monotonic() >= self.lease_settled else None
    if why is not None:
      if self.dormant:
        self.wake()
      if not self.open_link():
        return
      return self.spawn_worker(why)

    if self.transport is None:
      # nothing to do and nothing presented: only worth a bind if the far end
      # stays awake for it
      if not self.server_sleeps():
        self.hold()
      return
    if self.server_sleeps() and time.monotonic() - self.started >= DORMANT_HOLD:
      self.go_dormant()
    else:
      self.settle()

  def run(self) -> None:
    if not self.lender.start():
      gadget.log.error("jetlink: nothing can borrow the gadget from us; modeld will open it itself")
    try:
      while not self.stop:
        started = time.monotonic()
        try:
          self.step()
        except Exception:
          # nothing may escape: restarting in a loop is worse than sitting out a cycle
          gadget.log.exception("jetlink: unhandled error")
          self.close_link()
          self.next_attempt = time.monotonic() + RECONNECT_BACKOFF
        time.sleep(max(0.0, POLL - (time.monotonic() - started)))
    finally:
      # the sysctls stay: a stop here is where a drive begins
      self.lender.stop()
      self.stop_worker()
      self.close_link()
      gadget.set_dormant(False)
    gadget.log.warning("jetlink: stopped")


def main() -> None:
  gadget.set_logger(_log_to_file())
  owner = Owner()
  signal.signal(signal.SIGTERM, owner.request_stop)
  signal.signal(signal.SIGINT, owner.request_stop)
  owner.run()


if __name__ == "__main__":
  main()
