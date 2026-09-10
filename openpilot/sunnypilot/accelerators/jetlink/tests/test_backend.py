"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Holding the gadget while the Jetson comes up.

The comma is the USB device: the link exists only while a process holds ep0
with the UDC bound, and every unbind is an unplug the far end has to recover
from. This module is about not doing that, and about the single case that
still needs an edge.
"""

import unittest
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import backend


class FakeClock:
  """monotonic and sleep, so a 45 s wait costs no wall clock."""

  def __init__(self, now: float = 1000.0):
    self.now = now
    self.slept = 0.0

  def monotonic(self) -> float:
    return self.now

  def sleep(self, seconds: float) -> None:
    step = max(seconds, 0.01)
    self.now += step
    self.slept += step


class WaitForHost(unittest.TestCase):
  def setUp(self):
    self.clock = FakeClock()
    p = mock.patch.object(backend, 'time', self.clock)
    self.addCleanup(p.stop)
    p.start()
    self.client = mock.Mock()

  def bus(self, udc: str, cc: bool = True, attached: bool = False):
    for name, value in (('host_attached', attached), ('udc_state', udc), ('port_has_host', cc)):
      p = mock.patch.object(backend.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()

  def wait(self, seconds: float = backend.CONNECT_TIMEOUT) -> bool:
    return backend._wait_for_host(self.clock.monotonic() + seconds, self.client)

  def test_a_host_that_is_already_there_is_not_waited_for(self):
    self.bus('configured', attached=True)
    assert self.wait() is True
    assert self.clock.slept == 0.0

  def test_a_bus_that_stalls_half_enumerated_is_bounced_once(self):
    # A jetson whose hubs are not armed for remote wakeup answers the bind
    # with a bus reset and stops there; only another connect moves it.
    self.bus('default')
    assert self.wait() is False
    assert self.client.rebind.call_count == 1

  def test_the_bounce_waits_out_a_normal_enumeration(self):
    self.bus('addressed')
    backend._wait_for_host(self.clock.monotonic() + backend.STALLED_ENUMERATION / 2, self.client)
    assert self.client.rebind.call_count == 0

  def test_nothing_on_the_cable_is_not_a_stall(self):
    # No host on the CC pin: there is nobody to enumerate us and bouncing the
    # gadget would only cost the next one its bind.
    self.bus('not attached', cc=False)
    assert self.wait() is False
    assert self.client.rebind.call_count == 0

  def test_a_jetson_still_booting_is_left_alone(self):
    # Powered but not yet driving the bus: the UDC never leaves powered.
    self.bus('powered')
    assert self.wait() is False
    assert self.client.rebind.call_count == 0

  def test_a_bounce_that_fails_does_not_end_the_wait(self):
    self.bus('default')
    self.client.rebind.side_effect = OSError('no such device')
    assert self.wait() is False
    assert self.client.rebind.call_count == 1

  def test_a_host_that_turns_up_late_is_still_joined(self):
    states = ['powered'] * 3 + ['configured']
    attached = iter([False] * 3 + [True] * 5)
    with mock.patch.object(backend.helpers, 'host_attached', side_effect=lambda: next(attached)), \
         mock.patch.object(backend.helpers, 'udc_state', side_effect=lambda: states.pop(0) if states else 'configured'), \
         mock.patch.object(backend.helpers, 'port_has_host', return_value=True):
      assert self.wait() is True


class HoldingTheGadget(unittest.TestCase):
  """A link the attempt could not use goes back for the next one.

  Closing it unbinds the UDC, and while a Jetson boots the join loop asks
  again every few seconds: that was an unplug every cycle, and one of them
  landed on the enumeration.
  """

  def setUp(self):
    self.clock = FakeClock()
    p = mock.patch.object(backend, 'time', self.clock)
    self.addCleanup(p.stop)
    p.start()
    self.client = mock.Mock()
    for name, value in (('host_attached', False), ('udc_state', 'powered'), ('port_has_host', False)):
      p = mock.patch.object(backend.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()

  def test_a_link_nobody_enumerated_is_handed_back(self):
    held = []
    with self.assertRaises(TimeoutError):
      backend._connect_patiently(self.client, held.append)
    assert held == [self.client]
    assert self.client.close.call_count == 0, 'unbound the gadget between attempts'

  def test_without_somewhere_to_put_it_the_link_is_closed(self):
    # A FunctionFS owner that walks away wedges the driver until a reboot.
    with self.assertRaises(TimeoutError):
      backend._connect_patiently(self.client, None)
    assert self.client.close.call_count == 1

  def test_a_gadget_that_will_not_open_is_still_reported(self):
    # jetlinkd still holding ep0, or a gadget boot never created: there is no
    # link to hold on to and the join loop should hear why.
    with mock.patch.object(backend.helpers, 'connect', side_effect=OSError('ep0 busy')):
      with self.assertRaises(OSError):
        backend._connect_patiently(None, lambda c: self.fail('held a link that never opened'))

  def test_a_model_the_jetson_has_not_built_keeps_the_gadget_presented(self):
    # The panel says what is going on; a closed gadget would take the whole
    # link off the bus for the drive instead.
    held = []
    with mock.patch.object(backend.spec_cache, 'load', return_value=None):
      with self.assertRaises(RuntimeError):
        backend._open_link(self.client, hold=held.append)
    assert held == [self.client]
    assert self.client.close.call_count == 0


if __name__ == '__main__':
  unittest.main()
