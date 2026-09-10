"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The process that holds the gadget: what it keeps, what it lets go of, and when
it starts the heavy half.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import gadget, owner


class OwnerTest(unittest.TestCase):
  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.params = self.tmp / 'params'
    self.params.mkdir()
    self.write('JetlinkEnabled', b'1')
    self.write('IsOffroad', b'1')
    for name, value in (('DORMANT', self.tmp / 'dormant'),
                        ('SHUTDOWN_REQUEST', self.tmp / 'shutdown'),
                        ('STATE', self.tmp / 'state'),
                        ('params_dir', mock.Mock(return_value=self.params)),
                        ('link_configured', mock.Mock(return_value=True)),
                        ('link_endpoint', mock.Mock(return_value=None)),
                        ('can_setup_gadget', mock.Mock(return_value=False)),
                        ('host_attached', mock.Mock(return_value=True)),
                        ('udc_state', mock.Mock(return_value='configured')),
                        ('wait_for_host', mock.Mock(return_value=True))):
      p = mock.patch.object(gadget, name, value)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(owner, 'LOG', self.tmp / 'owner.log')
    self.addCleanup(p.stop)
    p.start()
    p = mock.patch.object(owner, 'vmtune', mock.Mock())
    self.addCleanup(p.stop)
    self.vmtune = p.start()

  def write(self, key: str, value: bytes) -> None:
    (self.params / key).write_bytes(value)

  def note_state(self, **kw) -> None:
    gadget.STATE.write_text(json.dumps(kw))

  def owner(self, presented=True, lendable=False):
    o = owner.Owner()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock(lendable=lendable) if presented else None
    for name in ('open_link', 'spawn_worker'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(o, 'close_link', mock.Mock(side_effect=lambda: setattr(o, 'transport', None)))
    self.addCleanup(p.stop)
    p.start()
    # a run has already reported, so nothing is outstanding and the far end sleeps
    self.note_state(sleep_after=1.0, unfinished=False)
    o.seen = o.marks()
    o.had_host = True
    return o


class TestOnroad(OwnerTest):
  """Once the car is moving the owner holds the gadget and stays off the bus."""

  def onroad(self) -> None:
    self.write('IsOffroad', b'0')

  def test_it_holds_the_gadget_and_does_nothing_else(self):
    o = self.owner(lendable=True)
    self.onroad()
    o.step()
    o.spawn_worker.assert_not_called()
    o.close_link.assert_not_called()
    o.transport.release_endpoints.assert_not_called()

  def test_a_run_still_going_is_stopped_so_modeld_can_borrow(self):
    # a build started while parked can still be running when the driver pulls
    # away; the lease it holds would keep modeld out for the whole drive
    o = self.owner(lendable=True)
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.onroad()
    o.step()
    worker.terminate.assert_called_once()

  def test_a_borrower_keeps_the_gadget_on_the_bus(self):
    o = self.owner(lendable=True)
    o.lender.lent = True
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_not_called()

  def test_endpoints_left_open_are_put_down_without_letting_go_of_ep0(self):
    o = self.owner(lendable=False)
    self.onroad()
    o.step()
    o.transport.release_endpoints.assert_called_once()
    o.close_link.assert_not_called()

  def test_a_borrower_wakes_a_dormant_owner(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.lender.lent = True
    o.step()
    self.assertFalse(o.dormant)
    o.open_link.assert_called_once()


class TestParked(OwnerTest):
  """Letting the Jetson sleep, and taking the gadget back when there is work."""

  def test_the_gadget_is_held_until_the_hold_has_passed(self):
    o = self.owner()
    o.step()
    self.assertFalse(o.dormant)

  def test_it_releases_once_there_is_nothing_to_do_and_the_far_end_sleeps(self):
    o = self.owner()
    o.started = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)
    o.close_link.assert_called_once()
    self.assertTrue(gadget.dormant())

  def test_a_far_end_that_never_sleeps_keeps_the_gadget(self):
    # on ignition power the Jetson stays up, and letting go would leave a
    # powered awake box unenumerated for the whole parked period
    o = self.owner()
    self.note_state(sleep_after=0.0, unfinished=False)
    o.started = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertFalse(o.dormant)
    o.close_link.assert_not_called()

  def test_a_far_end_too_old_to_say_keeps_the_release_it_always_had(self):
    gadget.STATE.unlink(missing_ok=True)
    o = self.owner()
    o.started = time.monotonic() - owner.DORMANT_HOLD
    o.step()
    self.assertTrue(o.dormant)


class TestStartingTheHeavyHalf(OwnerTest):
  """The owner cannot tell whether there is work: that needs the catalog, the
  spec and the Jetson. It notices what could have changed the answer."""

  def test_the_first_look_of_the_boot_always_runs(self):
    o = self.owner()
    o.seen = {}
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_new_pick_starts_a_run(self):
    o = self.owner()
    o.step()
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "b" * 40}')
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_jetson_turning_up_starts_a_run(self):
    o = self.owner()
    o.had_host = False
    o.step()
    o.spawn_worker.assert_called_once()

  def test_a_shutdown_request_starts_a_run(self):
    o = self.owner()
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))
    o.step()
    o.spawn_worker.assert_called_once()

  def test_an_unfinished_run_is_tried_again_on_its_own_timer(self):
    o = self.owner()
    self.note_state(sleep_after=1.0, unfinished=True)
    o.next_worker = time.monotonic() + owner.WORKER_BACKOFF
    o.step()
    o.spawn_worker.assert_not_called()
    o.next_worker = 0.0
    o.step()
    o.spawn_worker.assert_called_once()

  def test_only_one_run_at_a_time(self):
    o = self.owner()
    o.seen = {}
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_dormant_owner_wakes_before_starting_one(self):
    o = self.owner(presented=False)
    o.dormant = True
    o.seen = {}
    o.step()
    self.assertFalse(o.dormant)
    o.spawn_worker.assert_called_once()

  def test_nothing_is_started_over_the_servers_teardown(self):
    # the borrower let go a moment ago and the server is still reopening the
    # gadget it lost; a hello inside that window costs a re-enumeration
    o = self.owner()
    o.lender.lent = True
    o.step()
    o.lender.lent = False
    o.seen = {}
    o.step()
    o.spawn_worker.assert_not_called()
    o.lease_settled = 0.0
    o.step()
    o.spawn_worker.assert_called_once()


class TestTheRunThatFinishes(OwnerTest):
  def finished(self, o):
    """A worker that has just exited."""
    o.worker = mock.Mock(**{'poll.return_value': 0, 'returncode': 0})

  def test_a_successful_provision_does_not_start_a_second_run(self):
    # a run writes JetlinkSpec and JetlinkEngineReady itself, so a mark taken
    # when it was spawned always differs by the time it exits
    o = self.owner()
    self.finished(o)
    self.write('JetlinkEngineReady', b'a' * 64)
    self.write('JetlinkSpec', b'{}')
    o.step()
    o.spawn_worker.assert_not_called()

  def test_a_pick_changed_while_the_run_was_going_is_still_seen(self):
    o = self.owner()
    o.worker = mock.Mock(**{'poll.return_value': None})
    o.step()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "c"}')
    o.worker.poll.return_value = 0
    o.step()
    # the mark is retaken when the run exits, so this looks unchanged...
    o.spawn_worker.assert_not_called()
    self.write('ModelManager_ActiveBundleChestnut', b'{"ref": "d"}')
    o.step()
    o.spawn_worker.assert_called_once()   # ...and a later pick still starts one


class TestShutdown(OwnerTest):
  """hardwared waits 25 s and a build takes minutes, so the request cannot
  queue behind a provisioning run."""

  def request(self) -> None:
    gadget.SHUTDOWN_REQUEST.write_text(json.dumps({'reason': 'car battery'}))

  def test_it_does_not_wait_for_a_run_in_flight(self):
    o = self.owner()
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.request()
    o.step()
    worker.terminate.assert_called_once()
    o.spawn_worker.assert_called_once()
    assert 'shut down' in o.spawn_worker.call_args.args[0]

  def test_a_borrower_is_left_alone(self):
    # modeld has the endpoints: this cannot talk over it, and hardwared only
    # shuts a parked car down anyway
    o = self.owner()
    o.lender.lent = True
    self.request()
    o.step()
    o.spawn_worker.assert_not_called()


class TestNobodyCanBorrow(OwnerTest):
  def test_a_gadget_nobody_can_ask_for_is_given_to_the_drive(self):
    # holding ep0 with no way to lend it would keep modeld out for the whole
    # drive; without a lease it opens the gadget itself as it always did
    o = self.owner(lendable=True)
    o.lender.listening = False
    self.write('IsOffroad', b'0')
    o.step()
    o.close_link.assert_called_once()

  def test_parked_it_still_provisions(self):
    o = self.owner(lendable=True)
    o.lender.listening = False
    o.seen = {}
    o.step()
    o.close_link.assert_not_called()
    o.spawn_worker.assert_called_once()


class TestTheToggle(OwnerTest):
  def test_turning_it_off_lets_everything_go(self):
    o = self.owner()
    o.vm_tuned = True
    worker = o.worker = mock.Mock(**{'poll.return_value': None})
    self.write('JetlinkEnabled', b'0')
    o.step()
    o.close_link.assert_called_once()
    worker.terminate.assert_called_once()
    self.vmtune.restore_vm_tuning.assert_called_once()

  def test_the_sysctls_go_in_once_and_stay_for_the_drive(self):
    o = self.owner()
    o.step()
    o.step()
    self.vmtune.apply_vm_tuning.assert_called_once()
    self.vmtune.restore_vm_tuning.assert_not_called()

  def test_a_jetson_on_ethernet_has_no_gadget_to_own(self):
    with mock.patch.object(gadget, 'link_endpoint', return_value=('10.0.0.2', 5599)):
      o = owner.Owner()
      o.lender = mock.Mock(lent=False, listening=True)
      self.assertFalse(o.open_link())
      self.assertIsNone(o.transport)


class TestSetup(OwnerTest):
  def test_a_gadget_boot_did_not_make_is_created(self):
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'can_setup_gadget', return_value=True), \
         mock.patch.object(gadget, 'setup_gadget', return_value=True) as setup:
      self.assertTrue(o.ensure_gadget())
      setup.assert_called_once()

  def test_a_failed_setup_is_not_retried_every_cycle(self):
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False), \
         mock.patch.object(gadget, 'can_setup_gadget', return_value=True), \
         mock.patch.object(gadget, 'setup_gadget', return_value=False) as setup:
      for _ in range(3):
        o.ensure_gadget()
      setup.assert_called_once()

  def test_a_device_that_cannot_make_one_still_tries_the_link(self):
    o = self.owner(presented=False)
    with mock.patch.object(gadget, 'link_configured', return_value=False):
      self.assertTrue(o.ensure_gadget())


class TestLending(OwnerTest):
  def test_lendable_only_once_the_endpoints_are_down(self):
    o = self.owner(lendable=False)
    self.assertFalse(o.lendable())
    o.transport.lendable = True
    self.assertTrue(o.lendable())
    o.transport = None
    self.assertFalse(o.lendable())

  def test_a_stuck_write_is_freed_by_the_owner(self):
    o = self.owner()
    o.transport.rebind.return_value = True
    self.assertTrue(o.bounce_gadget())
    o.transport.rebind.assert_called_once()

  def test_nothing_to_bounce_is_not_an_error(self):
    o = self.owner(presented=False)
    self.assertFalse(o.bounce_gadget())
    self.assertFalse(o.lendable())

  def test_a_bounce_that_raises_is_not_an_error(self):
    o = self.owner()
    o.transport.rebind.side_effect = OSError('no such device')
    self.assertFalse(o.bounce_gadget())


if __name__ == '__main__':
  unittest.main()
