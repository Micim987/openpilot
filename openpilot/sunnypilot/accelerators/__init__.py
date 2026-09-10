"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

An accelerator that runs the large driving model off the comma: jetlink.

comma's chestnut board is not one of these: modeld, hardwared and the UI handle
it natively and only ask here when no board is fitted. Selection is
`if chestnut_present(): native elif accelerators.ready(): jetlink`.

Every function is a thin call into jetlink.backend and is safe on any device:
feature off costs a param read, package absent answers the negative default.
present(), ready(), progress() and enabled() are polled by the UI at 5 Hz and
must stay cheap.

Both models are the model manager's. The small one runs on whichever modeld
its bundle needs, and the accelerator joins that modeld. The big one is the
big-model slot, the same pick a chestnut runs from (the rule is with
ModelManagerSP._fetch_big_model_files). Nothing here changes which modeld
manager runs.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.accelerators.jetlink import backend

# written by jetlinkd and the joining state, read by the UI; a param because
# the writer is another process
P_PROGRESS = "AcceleratorProgress"
# Each report is a file write, and the UI reads it at 5 Hz. An upload reports
# once per 4 MB chunk, 440 of them for a 1.7 GB model, and onroad that is IO a
# recording would have to share the disk with.
PROGRESS_MIN_INTERVAL = 0.25
_last_progress = ('', 0.0)


class Daemon(NamedTuple):
  """A process the accelerator needs, for as long as the link is enabled.

  A description, not a PythonProcess: manager imports this package, so this
  package cannot import manager. process_config owns the onroad gating.
  """
  name: str
  module: str
  should_run: Callable[..., bool]


def installed() -> bool:
  """Is the backend's package checked out? What makes the link worth offering in the UI."""
  return backend.installed()


def present() -> bool:
  """Is a Jetson attached, or asleep and known to be there? USB-independent."""
  return backend.present()


def ready() -> bool:
  """Can the large model run right now? Params only, what the UI calls 'compiled'."""
  return backend.ready()


def unavailable_reason() -> str | None:
  """Why the link the user asked for cannot run, for the offroad alert. None unless enabled."""
  return backend.unavailable_reason()


def prepare() -> bool:
  """Process-wide setup modeld must do before going realtime, and a last veto. modeld only."""
  return backend.prepare()


def make_model_state(cam_w: int, cam_h: int, small=None):
  """The joining ModelState: the small model driving now, the Jetson swapped in later."""
  return backend.make_model_state(cam_w, cam_h, small)


def make_status_publisher(pm, model):
  """modeld's after_enqueue callback. Publishes nothing; logs telemetry at 1 Hz."""
  return backend.make_status_publisher(pm, model)


def enabled() -> bool:
  """Has the user turned the link on? Configuration only, never link state or ready()."""
  return backend.enabled()


def selected_model_name() -> str | None:
  """The big model the accelerator will run: the model manager's big-model pick, or the default."""
  return backend.selected_model_name()


def active_model_name() -> str | None:
  """selected_model_name() once the accelerator can run it, else None."""
  return backend.active_model_name()


def daemons() -> list[Daemon]:
  """Processes for process_config to build. They run onroad too: jetlinkd owns
  the USB gadget for as long as the link is enabled, and a gadget whose owner
  exits is an unplug the far end has to recover from."""
  return [Daemon("jetlinkd", "openpilot.sunnypilot.accelerators.jetlink.jetlinkd",
                 lambda started, params, CP: backend.enabled())]


def progress() -> dict | None:
  """{stage, frac, msg} while something provisions, else None.

  Read from the UI's param thread, so nothing may escape, UnknownKeyName included.
  """
  try:
    value = Params().get(P_PROGRESS)
  except Exception:
    return None
  return value if isinstance(value, dict) else None


def report_progress(stage: str, frac: float, msg: str = '') -> None:
  """Never raises: called from except handlers.

  Held to 4 Hz within a stage. The end of one always goes through, so the last
  thing the panel is told is never dropped.
  """
  global _last_progress
  last_stage, last_at = _last_progress
  now = time.monotonic()
  if frac < 1.0 and stage == last_stage and now - last_at < PROGRESS_MIN_INTERVAL:
    return
  _last_progress = (stage, now)
  try:
    Params().put(P_PROGRESS, {'stage': stage, 'frac': round(frac, 4), 'msg': msg})
  except Exception:
    cloudlog.exception("accelerators: could not report progress")


def clear_progress() -> None:
  try:
    Params().remove(P_PROGRESS)
  except Exception:
    cloudlog.exception("accelerators: could not clear progress")


def shutdown(reason: str = '', timeout: float = 25.0) -> None:
  """The device is powering off for good. Tell the Jetson, within `timeout`.

  hardwared calls this before DoShutdown and publishes no deviceState until it
  returns, so the request runs on a thread and is abandoned at the deadline.
  """
  if not backend.enabled():
    return

  def request():
    try:
      backend.shutdown(reason, timeout)
    except Exception:
      cloudlog.exception("accelerators: shutdown request failed")

  t = threading.Thread(target=request, name='accelerator-shutdown', daemon=True)
  t.start()
  t.join(timeout)
  if t.is_alive():
    cloudlog.warning("accelerators: shutdown request still pending after %.0f s, going on without it", timeout)
