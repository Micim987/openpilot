"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

modeld's per-frame status callback for the jetlink path.

On the chestnut path modeld hands `ChestnutState.send` to the model as
`after_enqueue`, called once the frame is on the accelerator and before the
result is waited for. This is the same hook. It publishes nothing: chestnutState
is comma's board on the wire, and a Jetson's telemetry has no message yet
(that needs a customReserved slot from the maintainers). Until then it goes to
swaglog at 1 Hz, which is enough to read temperature and power off a drive.

The hook still has to exist, because passing a callback is what makes the
client ask for telemetry: it is piggybacked on the previous inference response
(`want_state`), so it costs no extra round trip and cannot delay a frame.
"""
from __future__ import annotations

import time

from openpilot.common.swaglog import cloudlog

LOG_PERIOD = 1.0


class JetlinkStatus:
  def __init__(self, pm, model):
    self.pm = pm
    self.model = model
    # modeld's chestnut fallback clears this when it takes over; on the
    # jetlink path the joining state owns its own demotion and modeld re-raises
    # instead, so it stays true. Kept for the shared duck type.
    self.big = True
    self._last_logged = 0.0

  @property
  def client(self):
    # Per send rather than captured: a joining state has no client until the
    # Jetson turns up, and can lose and regain one without modeld hearing.
    return getattr(self.model, 'client', None)

  def send(self) -> None:
    client = self.client if self.big else None
    telemetry = client.last_state if client is not None else None
    if not telemetry:
      return
    now = time.monotonic()
    if now - self._last_logged < LOG_PERIOD:
      return
    self._last_logged = now
    cloudlog.event("jetlinkTelemetry", dead=bool(client.dead), **telemetry)
