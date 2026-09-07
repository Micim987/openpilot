"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The accelerator backend: everything core openpilot calls, and nothing else.

A module of functions behind sunnypilot.accelerators, which is the only thing
core openpilot imports. Holding the seam there is what keeps the patch against
upstream to a handful of lines - easy to review, easy to rebase across a sync,
and easy for another fork to lift. Anything only jetlinkd needs lives in
helpers or spec_cache, not here.

The `jetlink` client package can be absent on a device (a build that never
ran setup, or a fork that does not ship it). This module and helpers import
without it; the functions that need it import it when called and answer their
negative default when it is not there, logging once.
"""
from __future__ import annotations

import threading
import time

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.accelerators.jetlink import helpers, spec_cache

# How long one attempt holds the gadget open waiting for a host. This is no
# longer a deadline on the large model: make_model_state returns immediately and
# JoiningModelState keeps retrying for the life of the drive, because the Jetson
# is on the ignition rail and its boot starts after the comma is already onroad.
# Holding the gadget while we wait is the point - the Jetson sees us the moment
# it enumerates rather than on our next poll.
CONNECT_TIMEOUT = 45.0
CONNECT_DELAY = 0.5
# A serving accelerator gets ten frame periods to answer. Startup and engine
# loading use their own longer timeouts on the joining thread; a dead server
# must not hold the driving model's frame thread for several seconds.
INFERENCE_TIMEOUT = 0.5
# How long the load may wait for the early gadget bind. Sub-second when the
# endpoints are free; jetlinkd may still be letting go of them.
PRESENT_TIMEOUT = 5.0

# How long hardwared waits for jetlinkd to shut the Jetson down before the comma
# goes ahead without it. Wake from suspend is ~8 s to a server on the bench.
SHUTDOWN_TIMEOUT = 25.0

_missing_reported = False


def _package_missing(what: str) -> bool:
  """True, and logged once, if the jetlink client package cannot be imported."""
  global _missing_reported
  try:
    import jetlink.client  # noqa: F401
  except ImportError:
    if not _missing_reported:
      cloudlog.warning("jetlink: package not installed, %s unavailable", what)
      _missing_reported = True
    return True
  return False


def _wait_for_host(deadline: float) -> bool:
  reported = False
  while time.monotonic() < deadline:
    if helpers.host_attached():
      return True
    if not reported:
      cloudlog.warning("jetlink: gadget up, waiting for the jetson to enumerate")
      reported = True
    time.sleep(CONNECT_DELAY)
  return False


def _present_early(ready: dict) -> None:
  """Open and bind the gadget now, from a thread that is not modeld's.

  Not on the caller's thread: this runs from modeld's main thread, already
  SCHED_FIFO 54 pinned to core 7 by config_realtime_process, and the FunctionFS
  reader thread the open creates would inherit that in turn and preempt the
  frame loop for the drive (see joining._background_priority). A helper that
  drops realtime first is what the reader inherits from instead. Bounded, so a
  hung open cannot hold modeld's load; a helper that finishes after we stopped
  waiting closes what it opened rather than leaving the gadget held by nobody.
  """
  from openpilot.sunnypilot.accelerators.jetlink.joining import _background_priority
  lock = threading.Lock()

  deadline = time.monotonic() + PRESENT_TIMEOUT

  def present():
    _background_priority()
    # Retried, not attempted once. The one moment this runs is the moment
    # manager has just stopped jetlinkd to start modeld, and jetlinkd may not
    # have let go of ep0 yet: a single try loses the early bind in exactly the
    # case it exists for, and fails in milliseconds rather than using the
    # window. The caller stops waiting at the same deadline either way.
    client = None
    while client is None:
      try:
        client = helpers.connect()
      except Exception as e:
        if time.monotonic() >= deadline:
          cloudlog.warning("jetlink: could not present the gadget early (%s), the join will", e)
          return
        time.sleep(0.2)
    with lock:
      if ready.get('abandoned'):
        client.close()
      else:
        ready['client'] = client

  t = threading.Thread(target=present, name='jetlink-present', daemon=True)
  t.start()
  t.join(max(0.0, deadline - time.monotonic()) + 0.5)
  with lock:
    if t.is_alive():
      ready['abandoned'] = True
      cloudlog.warning("jetlink: presenting the gadget took over %.0f s, the join will", PRESENT_TIMEOUT)


def _connect_patiently(client=None):
  """Open the link, tolerating a busy gadget or a Jetson that is still booting.

  A `client` already holding the gadget (presented early, see make_model_state)
  skips the open and goes straight to waiting for the host."""
  deadline = time.monotonic() + CONNECT_TIMEOUT
  last = None
  while True:
    if client is None:
      try:
        client = helpers.connect()
      except Exception as e:
        client, last = None, e
    if client is not None:
      if helpers.host_attached():
        return client
      # We hold the gadget now, so the Jetson can see us the moment it is up.
      # Keep it open and wait rather than churning the endpoints.
      if _wait_for_host(deadline):
        return client
      client.close()
      raise TimeoutError(f"no jetson attached within {CONNECT_TIMEOUT:.0f}s")
    if time.monotonic() > deadline:
      raise last if last is not None else TimeoutError("could not open the link")
    cloudlog.warning("jetlink: link not ready (%s), retrying", last)
    time.sleep(CONNECT_DELAY)


def enabled() -> bool:
  return helpers.enabled()


def present() -> bool:
  return helpers.gadget_present()


def ready() -> bool:
  # Params only: no link IO, no file reads. Provisioning is jetlinkd's job,
  # so by the time modeld starts the answer is already recorded.
  if not helpers.enabled() or helpers.gadget_error() is not None:
    return False
  spec = spec_cache.load()
  selected = helpers.selected_model()
  return (spec is not None and selected is not None and spec.sha256 == selected['oid']
          and helpers.engine_ready_for(spec.sha256))


def unavailable_reason() -> str | None:
  return helpers.gadget_alert()


def prepare() -> bool:
  # Nothing to wait for on the link: make_model_state returns immediately and
  # joins in the background.
  #
  # The warp is the one thing worth refusing on, and it is not a race we can
  # join our way out of: scons builds it before manager starts, so a warp that
  # is missing when modeld starts stays missing for the whole drive. Saying no
  # here keeps modeld on the plain small model instead of leaving a joining
  # thread to retry something that cannot arrive. The constructor still
  # checks, against the geometry modeld actually has.
  if _package_missing('the large model'):
    return False
  from openpilot.sunnypilot.accelerators.jetlink import warp_cache
  if not warp_cache.is_cached(*warp_cache.device_geometry()):
    cloudlog.warning("jetlink: no warp compiled yet, staying on the small model")
    return False
  # The last hook that still runs before modeld goes SCHED_FIFO on core 7,
  # and the GPU's init spawns a thread that would inherit that. See
  # warp_cache.init_device.
  warp_cache.init_device()
  return True


def make_model_state(cam_w: int, cam_h: int, small=None):
  # Returns straight away with the small model in the driving seat. See
  # joining.py for why modeld must not be made to wait for a Jetson.
  if _package_missing('the large model'):
    return None
  from openpilot.sunnypilot.accelerators.jetlink.joining import JoiningModelState
  from openpilot.sunnypilot.accelerators.jetlink import warp_cache

  # The warp, loaded and warmed here rather than at the swap. This runs on
  # modeld's main thread inside make_model_state, before the frame loop
  # starts, which is the one moment the GPU is idle and there are no frames
  # to drop; the swap itself lands on the frame loop. Sized from the cached
  # spec, which is what the link will hand back, so the swap can use it
  # without a load. Another geometry is rejected; GPU compilation must not
  # be deferred to a driving frame.
  ready: dict = {}

  def prepare():
    from openpilot.sunnypilot.accelerators.jetlink.fallback import prepare_reset
    # The gadget first, so the Jetson enumerates and the server opens us
    # while the warp loads. Left to the join thread, the bind landed ~3 s
    # later than this on every ignition of the 2026-09-04 drives: that thread
    # starts as the small model runs its first frame, 1.3 s of tinygrad
    # holding the GIL, and the connect trails it. It also puts the bind, and
    # the USB enumeration it triggers, before any frame is in flight; one
    # ignition that day had a 655 ms small-model frame during the bind and
    # 17 s of modeldLagging for it. Failure here costs nothing: the join
    # thread opens the link itself if there is nothing to take over.
    _present_early(ready)
    cached = spec_cache.load()
    if cached is not None:
      img_h, img_w = cached.input_shapes['img'][2:]
      geometry = (img_w * 2, img_h * 2)
    else:
      geometry = warp_cache.device_geometry()[2:]
    try:
      ready['reset_small'] = prepare_reset(small)
      warp = warp_cache.load_warp(cam_w, cam_h, *geometry)
      warp_cache.warm(warp, cam_w, cam_h)
    except Exception:
      client = ready.pop('client', None)
      if client is not None:
        client.close()
      raise
    ready.update(warp=warp, geometry=geometry)

  def build(client, spec):
    from openpilot.sunnypilot.accelerators.jetlink.model_state import JetlinkModelState
    img_h, img_w = spec.input_shapes['img'][2:]
    warp = ready.get('warp') if ready.get('geometry') == (img_w * 2, img_h * 2) else None
    if warp is None:
      raise RuntimeError('no prepared warp for the server model geometry')
    return JetlinkModelState(cam_w, cam_h, client, spec, small, warp=warp)

  def connect():
    # The early client is good for one attempt: after that the join thread
    # opens its own, as it does for every rejoin.
    return _open_link(ready.pop('client', None))

  return JoiningModelState(cam_w, cam_h, small, connect, build, prepare,
                           reset_small=lambda: ready['reset_small']())


def _open_link(client=None):
  """Get a client and a spec. Link IO only, so it is safe off modeld's thread.

  Everything that touches tinygrad stays in `build`, on modeld's own thread:
  unpickling the warp JIT next to the small model running frames on the same
  device is not a race worth taking.
  """
  from jetlink.client import EngineMissing

  cached = spec_cache.load()
  if cached is None:
    if client is not None:
      client.close()
    raise RuntimeError("no cached jetlink model spec; jetlinkd has not provisioned")

  selected = helpers.selected_model()
  if selected is None or selected['oid'] != cached.sha256:
    if client is not None:
      client.close()
    raise RuntimeError('selected jetlink model has not been provisioned')

  # Two things can keep us waiting, and both resolve on their own: manager
  # stops jetlinkd and starts modeld in the same pass without waiting, so the
  # endpoints may still be held; and the Jetson may still be booting.
  client = _connect_patiently(client)
  try:
    hello = client.hello(timeout=10.0)
    cloudlog.warning("jetlink: %s trt %s, engine %s, loaded %s",
                     hello.get('device'), hello.get('trt_version'),
                     hello.get('engine_state'), str(hello.get('loaded'))[:16])
    # jetlinkd left the engine loaded on the server, so this is normally one
    # round trip. If the server restarted it is a load from the plan cache,
    # 13 to 25 s measured, which the timeout has to cover.
    try:
      spec = client.ensure_engine(cached.sha256, cached.nbytes, frame_skip=cached.frame_skip,
                                  build_timeout=120.0)
    except EngineMissing:
      # The Jetson's cache was pruned, re-flashed or swapped since jetlinkd
      # recorded it as ready. Clear the record so jetlinkd provisions again
      # next time the car is parked, instead of every drive failing here.
      helpers.set_engine_ready(None)
      raise
    client.deadline = INFERENCE_TIMEOUT
    return client, spec
  except BaseException:
    client.close()
    raise


def make_status_publisher(pm, model):
  from openpilot.sunnypilot.accelerators.jetlink.status import JetlinkStatus
  # The model, not its client: with a joining state the link arrives after
  # modeld has already built this, and may go away and come back mid-drive.
  # Reading `model.client` per send is what lets the telemetry follow it.
  return JetlinkStatus(pm, model)


def uses_stock_runner() -> bool:
  # The toggle is the configuration. The model is not: JetlinkModel is a
  # choice that defaults through selected_model(), so an enabled device with
  # no model set still provisions and reports ready(). Gating on the model too
  # left that device on modeld_tinygrad under a custom small bundle, where
  # jetlink never runs and modelV2.big never happens. Not live presence or
  # readiness either: a late boot must not change which modeld manager runs
  # in the middle of a drive, and installing the package must not route
  # manager away from a custom small bundle on its own.
  return helpers.enabled()


def model_choices() -> list[dict]:
  if not helpers.link_configured():
    return []
  selected = helpers.selected_model() or {}
  cached = helpers._get('JetlinkCachedModels') or []
  return [{'name': m['name'], 'selected': m['oid'] == selected.get('oid'),
           'cached': m['oid'] in cached} for m in helpers.model_index()]


def select_model(name: str) -> None:
  from openpilot.common.params import Params
  if name not in {m['name'] for m in helpers.model_index()}:
    raise ValueError(f'unknown jetlink model: {name}')
  Params().put(helpers.P_MODEL, name)
  Params().put_bool(helpers.P_ENABLED, True)
  Params().remove('ModelRunnerTypeCache')


def active_model_name() -> str | None:
  if not ready():
    return None
  return next((m['name'] for m in model_choices() if m['selected']), None)


def shutdown(reason: str, timeout: float = SHUTDOWN_TIMEOUT) -> None:
  """Take the Jetson down with us. Runs in hardwared, so it cannot touch the
  link itself: jetlinkd owns the gadget offroad, and a Jetson that is asleep
  has to be woken by presenting it, which only the owner can do. Hand the
  request over and wait; jetlinkd needs ~10 s for the wake and one round
  trip, and manager will not kill it until after we return.

  Skipped when no Jetson is known to be there, dormant counts as there. Not
  skipped when jetlinkd is busy in a long provision: it will not see the
  request in time, and the timeout is what covers that.
  """
  if not helpers.enabled() or not helpers.gadget_present():
    return
  cloudlog.warning("jetlink: asking the jetson to power off: %s", reason)
  if not helpers.request_shutdown(reason):
    return
  if helpers.await_shutdown(timeout):
    cloudlog.warning("jetlink: shutdown request handed to the jetson")
  else:
    cloudlog.warning("jetlink: nobody took the shutdown request within %.0f s", timeout)
