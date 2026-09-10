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

  def test_no_model_picked_yet_keeps_the_gadget_presented(self):
    # The panel says what is going on; a closed gadget would take the whole
    # link off the bus for the drive instead.
    held = []
    with mock.patch.object(backend.helpers, 'selected_model', return_value=None):
      with self.assertRaises(RuntimeError):
        backend._open_link(self.client, hold=held.append)
    assert held == [self.client]
    assert self.client.close.call_count == 0


class BuildingOnroad(unittest.TestCase):
  """The picked model is built with the small model driving.

  jetlinkd provisions offroad only, so a model picked in the driveway and
  driven off on used to cost the whole drive: modeld would not even present
  the gadget, and the panel said a device was on the USB port.
  """

  ENTRY = {'name': 'CTM v2', 'ref': 'f' * 40, 'oid': 'a' * 64, 'size': 766 << 20}

  def setUp(self):
    from openpilot.sunnypilot.accelerators.jetlink import provision
    self.provision = provision
    self.client = mock.Mock()
    self.spec = mock.Mock(sha256=self.ENTRY['oid'])
    p = mock.patch.object(backend, '_connect_patiently', return_value=self.client)
    self.addCleanup(p.stop)
    p.start()
    for name, value in (('selected_model', dict(self.ENTRY)), ('shipped_model_path', None),
                        ('engine_ready_for', False)):
      p = mock.patch.object(backend.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(provision, 'ensure', return_value=self.spec)
    self.addCleanup(p.stop)
    self.ensure = p.start()

  def test_the_model_the_picker_names_is_what_gets_built(self):
    client, spec = backend._open_link()
    assert spec is self.spec and client is self.client
    assert self.ensure.call_args.args[1:3] == (self.ENTRY['oid'], self.ENTRY['size'])

  def test_the_frame_deadline_is_set_before_the_link_is_handed_over(self):
    # ensure_engine waits minutes; the frame path must not inherit that
    client, _ = backend._open_link()
    assert client.deadline == backend.INFERENCE_TIMEOUT

  def test_the_join_thread_can_be_stopped_through_the_build(self):
    stop = object()
    backend._open_link(should_stop=stop)
    assert self.ensure.call_args.kwargs['should_stop'] is stop

  def test_progress_reaches_the_panel(self):
    backend._open_link()
    assert self.ensure.call_args.kwargs['progress'] is self.provision.report_with_eta

  def test_bytes_neither_end_has_are_a_parked_job(self):
    # Downloading a gigabyte is the one part of provisioning that needs the
    # internet, and it is not something to start mid-drive.
    from jetlink.client import EngineMissing
    self.ensure.side_effect = EngineMissing('no engine')
    with mock.patch.object(backend.helpers, 'set_engine_ready') as cleared:
      with self.assertRaises(EngineMissing):
        backend._open_link()
    cleared.assert_called_once_with(None)
    self.client.close.assert_called_once()



class BorrowingTheGadget(unittest.TestCase):
  """modeld does not bring the gadget up any more.

  jetlinkd holds ep0 and the bind for as long as the link is enabled, so the
  comma stays enumerated across the ignition edge; modeld asks for the endpoint
  files and gives them back by exiting.
  """

  def test_the_loan_is_what_the_link_is_opened_over(self):
    loan = object()
    with mock.patch.object(backend, '_gadget_loan', return_value=loan), \
         mock.patch.object(backend.helpers, 'connect') as connect, \
         mock.patch.object(backend.helpers, 'host_attached', return_value=True):
      backend._connect_patiently()
    assert connect.call_args.kwargs['loan'] is loan
    assert connect.call_args.kwargs['name'] == 'modeld'

  def test_no_daemon_to_ask_still_opens_the_gadget(self):
    # the link was only just turned on, or jetlinkd died: a drive must not lose
    # the large model to a daemon fault
    from openpilot.sunnypilot.accelerators.jetlink import lending
    with mock.patch.object(lending, 'borrow', return_value=None):
      assert backend._gadget_loan() is None

  def test_a_daemon_that_throws_is_not_a_lost_drive(self):
    from openpilot.sunnypilot.accelerators.jetlink import lending
    with mock.patch.object(lending, 'borrow', side_effect=OSError('no socket')):
      assert backend._gadget_loan() is None


if __name__ == '__main__':
  unittest.main()
