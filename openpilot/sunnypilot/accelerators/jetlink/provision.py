"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Making the Jetson ready to run the model that is picked.

Both owners of the link do this. jetlinkd does it offroad, where it can spend
minutes downloading a gigabyte; modeld's join thread does it onroad, where it
cannot download but can upload a file the comma already has and then wait out
the build with the small model driving. What they share is here: the model's
identity, the upload nobody may make without proving the hash first, the
estimate that turns "build 12%" into a number of minutes, and the record of
what the server ended up with.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path

from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot import accelerators
from openpilot.sunnypilot.accelerators.jetlink import helpers, spec_cache

# An upload and a build on a busy box. The build itself is 102 to 294 s on this
# hardware; the ceiling is only there so a server that has stopped answering
# does not hold the caller for the rest of the day.
BUILD_TIMEOUT = 1800.0
# Progress is a param write, and the upload reports one per 4 MB chunk: a
# 1.7 GB model is 440 of them. The server throttles its own to 4 Hz; this
# holds the whole stream to the same rate, which onroad is IO the recording
# does not have to share the disk with.
REPORT_MIN_INTERVAL = 0.25
_last_report = 0.0


def identity(entry: dict) -> tuple[str, int]:
  """The picked model's sha256 and byte count.

  From the catalog model's LFS pointer, so the comma can name the model
  without holding or hashing the ONNX. The lookup happens once per model ever
  and is kept in a param; it needs the internet, and that is the only part of
  provisioning that does.
  """
  sha256, nbytes = entry.get('oid'), entry.get('size')
  if sha256 and nbytes:
    return sha256, int(nbytes)
  accelerators.report_progress('connect', 0.0, 'looking up the model')
  return helpers.resolve_pointer(entry['ref'])


def verified_upload(model_path: Path | None, sha256: str, nbytes: int) -> Path | None:
  """The file to upload, once its hash is proven to match the registry.

  Uploading under a sha the bytes do not have would leave the Jetson with a
  plan whose name lies about its contents, so the hash happens here, on the
  one path where the bytes go somewhere.
  """
  if model_path is None:
    return None
  try:
    if model_path.stat().st_size != nbytes:
      cloudlog.error("jetlink: %s is %d bytes, the registry says %d; not uploading it",
                     model_path.name, model_path.stat().st_size, nbytes)
      return None
    from jetlink.spec import sha256_file
    have, _ = sha256_file(str(model_path))
    if have != sha256:
      cloudlog.error("jetlink: %s hashes to %s, the registry says %s; not uploading it",
                     model_path.name, have[:16], sha256[:16])
      return None
  except OSError:
    return None
  return model_path


def ensure(client, sha256: str, nbytes: int, model_path: Path | None, *,
           progress=None, should_stop=None, build_timeout: float = BUILD_TIMEOUT):
  """Make the server ready for this model and remember what it answered.

  Asks without the file first: the server answers from the sha alone when it
  already has the model, which is every poll of a parked car and every join of
  a drive. EngineMissing means the Jetson has neither the plan nor the bytes
  and neither has the caller.
  """
  from jetlink.client import EngineMissing
  ask = functools.partial(client.ensure_engine, sha256, nbytes, progress=progress,
                          build_timeout=build_timeout, should_stop=should_stop)
  try:
    spec = ask(onnx_path=None)
  except EngineMissing:
    upload = verified_upload(model_path, sha256, nbytes)
    if upload is None:
      raise
    spec = ask(onnx_path=upload)
  spec_cache.store(spec, model_path)
  helpers.set_engine_ready(spec.sha256)
  return spec


def estimated_build_seconds(size: int) -> int:
  """Orin Nano Super, TensorRT 10.3: the 766 MB models built in 102 to 166 s,
  the 1.75 GB ones in 230 to 294 s."""
  return int(60 + 130 * size / 1e9)


def _eta(seconds: float) -> str:
  if seconds >= 90:
    return f"about {seconds / 60:.0f} min left"
  return f"about {max(seconds, 1):.0f}s left"


def report_with_eta(stage: str, frac: float, msg: str = '') -> None:
  """Progress, with how long the build still has to run.

  Estimated from the model's size, on measurements of this hardware. It
  belongs here rather than in the UI, which knows nothing about jetlink. Only
  the build is estimated: the upload reports MB of MB and a connect has
  nothing to predict.
  """
  global _last_report
  now = time.monotonic()
  if frac < 1.0 and now - _last_report < REPORT_MIN_INTERVAL:
    return
  _last_report = now
  if stage == 'build':
    size = (helpers.selected_model() or {}).get('size')
    if size:
      msg = _eta(estimated_build_seconds(size) * max(0.0, 1.0 - frac))
  accelerators.report_progress(stage, frac, msg)
