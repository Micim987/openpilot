#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Provisions whatever large model is selected, then exits.

A download, an upload and a TensorRT build take minutes, so this runs offroad,
and it is the heavy half of jetlink: numpy, the client, tinygrad for the warp.
That is why it is a run and not a daemon. owner.py holds the gadget for the
whole time the link is enabled and starts one of these when something changes;
this borrows the endpoint files from it exactly as modeld does (lending.py), so
the gadget never leaves the bus and a parked car keeps one resident jetlink
process of about 13 MB instead of this one's 47.

Without an owner to borrow from, this opens the gadget itself, as it always
did: a device whose owner died still provisions.

The result is cached on the Jetson, recorded in a param and left loaded on the
server. modeld does its own provisioning over its borrowed link when the picked
model turns out not to be built.

The owner stops this at the onroad transition with SIGTERM, so every long wait
polls `stop`; the server's build thread carries on regardless and modeld picks
the engine up over its own link.
"""
from __future__ import annotations

import json
import signal
import threading

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot import accelerators
from openpilot.common.params import Params
from openpilot.sunnypilot.accelerators.jetlink import helpers, lending, owner, provision, spec_cache, warp_cache

# how long to wait for the Jetson to enumerate before giving up on this run.
# The owner presented the gadget; a box that is asleep answers the bind in
# about 8 s, one that is off never does and the next run will find it
WAKE_TIMEOUT = 20.0

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
    self.fetch_failed = False
    self.verified = False   # the server has confirmed the ready param this run
    self.warp_built = False  # tried the comma-side warp this run
    self.warp_thread: threading.Thread | None = None
    self.loan = None        # the owner's lease on the endpoint files
    # does the far end suspend when the gadget goes? From the server's hello.
    # The owner needs it to decide whether letting go is worth what it costs,
    # and cannot ask: it never speaks the protocol
    self.server_sleeps = True

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
    """Should a long wait give up? Only a stop, now: the owner sends one at the
    onroad transition so modeld can take the endpoints.

    A build takes minutes, so one started while parked can still be running
    when the driver pulls away. The server's build thread carries on either way
    and modeld picks the engine up over its own link, so letting go costs
    nothing.
    """
    return self.stop






  def open_link(self) -> bool:
    """Borrow the endpoints from the owner, or open the gadget ourselves.

    A loan is the ordinary case. None means nobody is listening, which is a
    device whose owner died or one where the link was only just turned on; the
    gadget is ours to open then, as it was before there was an owner.
    """
    if self.client is not None:
      return True
    try:
      self.loan = lending.borrow('jetlinkd')
      self.client = helpers.connect(deadline=5.0, name='jetlinkd', loan=self.loan)
      if self.loan is None:
        cloudlog.warning("jetlink: no owner to borrow from, presenting the gadget ourselves")
      return True
    except Exception:
      cloudlog.exception("jetlink: could not open the link")
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
      if not self.open_link():
        raise RuntimeError("could not open the link")
      if not helpers.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce,
                                   should_stop=lambda: self.stop):
        raise TimeoutError(f"no jetson attached within {WAKE_TIMEOUT:.0f} s")
      resp = self.client.shutdown(reason, timeout=5.0)
      cloudlog.warning("jetlink: jetson answered the shutdown request: %s", resp)
    except Exception:
      cloudlog.exception("jetlink: could not shut the jetson down")
    finally:
      helpers.finish_shutdown()

  # -- one run --------------------------------------------------------------

  def bounce(self) -> bool:
    """Ask whoever owns the gadget to bounce it. Ours to do only when we opened
    it ourselves; otherwise the owner does it for us over the lease."""
    try:
      return bool(self.client.rebind()) if self.client is not None else False
    except Exception:
      cloudlog.exception("jetlink: could not bounce the gadget")
      return False

  def note_state(self, unfinished: bool) -> None:
    """What the owner cannot work out for itself: whether the far end sleeps
    when the gadget goes, and whether this run left anything undone."""
    try:
      owner.STATE.write_text(json.dumps({
        'sleep_after': 1.0 if self.server_sleeps else 0.0,
        'unfinished': unfinished,
      }))
    except OSError:
      cloudlog.exception("jetlink: could not record what the owner needs")

  def run(self) -> bool:
    """One provisioning round. True when there is nothing left to do."""
    if not helpers.enabled():
      return True
    try:
      helpers.migrate_selection()
    except Exception:
      cloudlog.exception("jetlink: could not migrate the model selection")
    # neither the link nor the Jetson is needed for this, and modeld will not
    # start the large model without it
    self.build_warp()

    reason = helpers.pending_shutdown()
    if reason is not None:
      self.shutdown_jetson(reason)
      return True

    if not self.has_work():
      cloudlog.warning("jetlink: nothing to provision")
      self.note_state(unfinished=False)
      return True

    finished = False
    try:
      if not self.open_link():
        return False
      if not helpers.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce,
                                   should_stop=self.interrupted):
        cloudlog.warning("jetlink: no jetson within %.0f s, leaving it for the next run", WAKE_TIMEOUT)
        return False
      finished = self.provision()
    except Exception:
      cloudlog.exception("jetlink: provisioning failed")
      accelerators.report_progress('failed', 1.0, 'see the log')
    finally:
      self.note_state(unfinished=not finished)
      self.close_link()
      if self.warp_thread is not None and self.warp_thread.is_alive():
        cloudlog.warning("jetlink: leaving with the warp still compiling; it will rebuild next run")
    return finished


def main() -> None:
  d = Jetlinkd()
  # the owner stops this at the onroad transition; closing the link properly is
  # what keeps the driver healthy for modeld
  signal.signal(signal.SIGTERM, d.request_stop)
  signal.signal(signal.SIGINT, d.request_stop)
  d.run()


if __name__ == "__main__":
  main()
