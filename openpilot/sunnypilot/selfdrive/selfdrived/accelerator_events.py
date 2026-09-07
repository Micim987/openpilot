"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Onroad events for an accelerator that joins mid-drive.

selfdrived's native big model block is written for a board that is loaded
before modelV2 is first published and is either running or failed from then
on. An accelerator on its own power (sunnypilot/accelerators) arrives whenever
it arrives, joins onto a modelV2 the small model is already publishing, and
can leave and come back. Everything it needs to say goes through modelV2.big
and the additive modelDataV2SP fields; nothing here reads a param.
"""
import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom
from openpilot.selfdrive.selfdrived.events import Events, EventName
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

EventNameSP = custom.OnroadEventSP.EventName
AcceleratorState = custom.ModelDataV2SP.AcceleratorState


class AcceleratorEvents:
  def __init__(self):
    self.big_model_available = False
    self.big_model_running = False

  def update(self, sm: messaging.SubMaster, enabled: bool, events: Events, events_sp: EventsSP) -> None:
    status = sm['modelDataV2SP']

    # Ignore missing/stale status without rearming the chime. Only a fresh
    # unavailable state (or successful activation) permits another announcement.
    if all(sm.seen[s] and sm.alive[s] and sm.valid[s] for s in ('modelV2', 'modelDataV2SP')):
      available = status.bigModelAvailable and not sm['modelV2'].big
      if available and not self.big_model_available:
        events_sp.add(EventNameSP.bigModelAvailable)
      self.big_model_available = available

    # A join that holds modelV2 back keeps the driver out, as the native load
    # does. A late join onto a model already publishing never blocks.
    if status.acceleratorState == AcceleratorState.joining and not sm.alive['modelV2']:
      events.add(EventName.bigModelLoading)

    # A fall while engaged is a soft disable: the plan just went from ~200 m to
    # ~5 m under the driver. Disengaged, the small model simply carries on.
    # Only counted for our accelerator: a chestnut that drops modelV2.big has
    # the native bigModelFailed for it, and this must not double that alert.
    running_big = sm.alive['modelV2'] and sm.valid['modelV2'] and sm['modelV2'].big and \
      status.acceleratorState != AcceleratorState.none
    if self.big_model_running and not running_big and enabled:
      events_sp.add(EventNameSP.bigModelLinkLost)
    self.big_model_running = running_big
