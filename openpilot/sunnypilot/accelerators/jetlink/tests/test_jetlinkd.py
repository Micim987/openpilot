"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What jetlinkd does when the far end is attached but not serving.

That is the expensive state, not the one where no Jetson is plugged in: the
daemon has a host to talk to and keeps trying, so anything it repeats per
attempt it repeats for as long as the car is parked.
"""

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import jetlinkd, provision


class FakeSpec:
  def __init__(self, sha256: str = 'deadbeef', nbytes: int = 1 << 20):
    self.sha256 = sha256
    self.nbytes = nbytes


class FakeSpecCache:
  """spec_cache backed by memory rather than a param."""

  def __init__(self):
    self.spec = None
    self.src = None
    self.stores = 0

  def load(self):
    return self.spec

  def source(self):
    return self.src

  def store(self, spec, source: Path | None = None) -> None:
    self.stores += 1
    self.spec = spec
    if source is not None:
      st = source.stat()
      self.src = (str(source), st.st_mtime_ns, st.st_size)


def fake_jetlink_spec_module(counter: list):
  """A stand-in for jetlink.spec, which is not importable without the package."""
  mod = types.ModuleType('jetlink.spec')

  def sha256_file(path, *a, **kw):
    counter.append(path)
    return 'deadbeef', 1 << 20

  mod.sha256_file = sha256_file

  client = types.ModuleType('jetlink.client')

  class EngineMissing(Exception):
    pass

  client.EngineMissing = EngineMissing
  return {'jetlink': types.ModuleType('jetlink'), 'jetlink.spec': mod, 'jetlink.client': client}


def serving_client(spec=None):
  """A client whose server already has the engine."""
  client = mock.Mock()
  client.ensure_engine.return_value = spec or FakeSpec()
  return client


class TestProvisionCost(unittest.TestCase):
  """What provisioning is allowed to cost when nothing needs doing.

  The identity comes from the catalog's pointer, so a parked car asks the Jetson what it
  already has without reading, hashing or even having the ONNX.
  """

  ENTRY = {'name': 'Fake', 'ref': 'f' * 40, 'oid': 'deadbeef', 'size': 4096}

  def setUp(self):
    self.model = Path(tempfile.mkdtemp()) / 'big_driving_supercombo.onnx'
    self.model.write_bytes(b'x' * 4096)
    self.cache = FakeSpecCache()
    self.hashed: list[str] = []
    p = mock.patch.object(jetlinkd, 'Params')
    self.addCleanup(p.stop)
    p.start()

    # the provisioning itself lives in provision.py, which jetlinkd and
    # modeld's join thread both call; both modules' references are stood in for
    for module in (jetlinkd, jetlinkd.provision):
      for target, new in (('spec_cache', self.cache), ('accelerators', mock.Mock())):
        p = mock.patch.object(module, target, new)
        self.addCleanup(p.stop)
        p.start()
    for name, value in (('shipped_model_path', self.model), ('engine_ready_for', False),
                        ('selected_model', dict(self.ENTRY))):
      p = mock.patch.object(jetlinkd.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.dict(sys.modules, fake_jetlink_spec_module(self.hashed))
    self.addCleanup(p.stop)
    p.start()

  def test_the_identity_comes_from_the_registry_not_the_file(self):
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      assert d.provision() is True
    args = d.client.ensure_engine.call_args.args
    assert args[0] == self.ENTRY['oid'] and args[1] == self.ENTRY['size']

  def test_a_model_asked_for_the_first_time_has_its_pointer_looked_up(self):
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'selected_model', return_value={**self.ENTRY, 'oid': None, 'size': None}), \
         mock.patch.object(jetlinkd.helpers, 'resolve_pointer', return_value=('deadbeef', 4096)) as resolve, \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      assert d.provision() is True
    resolve.assert_called_once_with('f' * 40)
    args = d.client.ensure_engine.call_args.args
    assert args[0] == 'deadbeef' and args[1] == 4096

  def test_a_pointer_that_cannot_be_looked_up_is_a_failed_provision(self):
    # the ordinary failure path: logged, backed off, tried again next poll
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'selected_model', return_value={**self.ENTRY, 'oid': None, 'size': None}), \
         mock.patch.object(jetlinkd.helpers, 'resolve_pointer', side_effect=OSError('offline')), \
         self.assertRaises(OSError):
      d.provision()
    d.client.ensure_engine.assert_not_called()

  def test_a_server_that_already_has_it_never_reads_the_file(self):
    # the steady state of a parked car: the 766 MB hash is not paid per retry
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    d.client.ensure_engine.return_value = FakeSpec()
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      for _ in range(3):
        d.verified = False
        assert d.provision() is True
    assert self.hashed == [], "hashed the model to ask a question the registry answers"

  def test_it_asks_even_with_no_model_on_disk(self):
    # The Jetson keeps its own copy of every ONNX and never prunes them, so a
    # comma that has deleted its own can still use an engine already built.
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'shipped_model_path', return_value=None), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      assert d.provision() is True
    assert d.client.ensure_engine.call_args.kwargs['onnx_path'] is None

  def test_a_server_that_wants_the_bytes_gets_them_fetched(self):
    from jetlink.client import EngineMissing
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    d.client.ensure_engine.side_effect = EngineMissing('no engine')
    with mock.patch.object(jetlinkd.helpers, 'shipped_model_path', return_value=None), \
         mock.patch.object(d, 'fetch_model', return_value=self.model) as fetch:
      # False, not an exception: the download takes minutes and the link is
      # not held through it; the next poll tries again.
      assert d.provision() is False
    fetch.assert_called_once()

  def _wants_the_bytes(self):
    from jetlink.client import EngineMissing
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    d.client.ensure_engine.side_effect = EngineMissing('no engine')
    return d, EngineMissing

  def test_a_file_that_is_not_the_registry_model_is_never_uploaded(self):
    # Trusting the pointer for the identity is right for asking and wrong for
    # answering: uploading under a sha the bytes do not have would leave the
    # Jetson with a plan whose name lies about its contents.
    d, EngineMissing = self._wants_the_bytes()
    with mock.patch.object(jetlinkd.helpers, 'selected_model',
                           return_value={**self.ENTRY, 'oid': 'not-what-the-file-hashes-to'}), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'), \
         self.assertRaises(EngineMissing):
      d.provision()
    assert all(c.kwargs['onnx_path'] is None for c in d.client.ensure_engine.call_args_list)

  def test_a_file_of_the_wrong_size_is_never_uploaded(self):
    d, EngineMissing = self._wants_the_bytes()
    with mock.patch.object(jetlinkd.helpers, 'selected_model',
                           return_value={**self.ENTRY, 'size': 999999}), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'), \
         self.assertRaises(EngineMissing):
      d.provision()
    assert all(c.kwargs['onnx_path'] is None for c in d.client.ensure_engine.call_args_list)
    assert self.hashed == [], "size is the cheap check and comes first"

  def test_the_file_is_uploaded_once_it_is_proven_to_be_the_model(self):
    d, _ = self._wants_the_bytes()
    d.client.ensure_engine.side_effect = [d.client.ensure_engine.side_effect, FakeSpec()]
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      assert d.provision() is True
    calls = d.client.ensure_engine.call_args_list
    assert calls[0].kwargs['onnx_path'] is None, "asked without the file first"
    assert calls[1].kwargs['onnx_path'] == self.model
    assert self.hashed == [str(self.model)], "hashed once, on the path the bytes leave by"

  def test_a_ready_engine_short_circuits_without_touching_the_file(self):
    self.cache.store(FakeSpec(), self.model)
    d = jetlinkd.Jetlinkd()
    d.verified = True
    with mock.patch.object(jetlinkd.helpers, 'engine_ready_for', return_value=True):
      assert d.provision() is True
    assert self.hashed == []

  def test_a_ready_param_is_checked_with_the_server_once_per_attach(self):
    # the Jetson's cache can be pruned or re-flashed under a param that says ready
    self.cache.store(FakeSpec(), self.model)
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'engine_ready_for', return_value=True), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready') as ready:
      assert d.provision() is True
      assert d.client.ensure_engine.call_count == 1
      assert d.verified
      assert d.provision() is True
      assert d.client.ensure_engine.call_count == 1, "verified once, then the param is trusted"
    ready.assert_called_with('deadbeef')

  def test_the_shapes_come_from_the_server_not_the_file(self):
    d = jetlinkd.Jetlinkd()
    d.client = serving_client(FakeSpec(sha256='deadbeef', nbytes=1 << 20))
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      assert d.provision() is True
    d.client.ensure_engine.assert_called_once()
    kwargs = d.client.ensure_engine.call_args.kwargs
    assert callable(kwargs['should_stop'])
    assert self.cache.stores == 1 and self.cache.spec.sha256 == 'deadbeef'

  def test_stop_is_polled_through_the_long_wait(self):
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      d.provision()
    should_stop = d.client.ensure_engine.call_args.kwargs['should_stop']
    assert should_stop() is False
    d.request_stop()
    assert should_stop() is True


class TestBackoff(unittest.TestCase):
  def test_it_doubles_and_then_stops(self):
    d = jetlinkd.Jetlinkd()
    seen = []
    for _ in range(8):
      d.failures += 1
      seen.append(d.backoff())
    assert seen[:5] == [30.0, 60.0, 120.0, 240.0, 480.0]
    assert seen[-1] == jetlinkd.RETRY_BACKOFF_MAX


class TestStepOnFailure(unittest.TestCase):
  def _step(self, d, provision):
    for p in (mock.patch.object(jetlinkd, 'accelerators', mock.Mock()),
              mock.patch.object(jetlinkd.helpers, 'enabled', return_value=True),
              mock.patch.object(jetlinkd.helpers, 'host_attached', return_value=True),
              mock.patch.object(d, 'open_link', return_value=True),
              mock.patch.object(d, 'provision', provision),
              mock.patch.object(d, 'close_link', mock.Mock())):
      self.addCleanup(p.stop)
      p.start()
    d.step()

  def test_a_timeout_keeps_the_gadget_presented(self):
    # unbinding makes the host re-enumerate; on every retry that was hours of
    # connect/disconnect against a Jetson not running the server
    d = jetlinkd.Jetlinkd()
    d.client = object()
    with mock.patch.object(jetlinkd, '_timed_out', return_value=True):
      self._step(d, mock.Mock(side_effect=RuntimeError('no reply')))
    assert d.close_link.call_count == 0
    assert d.failures == 1
    assert d.next_provision > time.monotonic()

  def test_anything_else_reopens_the_link(self):
    d = jetlinkd.Jetlinkd()
    d.client = object()
    with mock.patch.object(jetlinkd, '_timed_out', return_value=False):
      self._step(d, mock.Mock(side_effect=RuntimeError('desynced')))
    assert d.close_link.call_count == 1
    assert d.failures == 1

  def test_a_host_that_arrives_does_not_serve_out_the_old_backoff(self):
    d = jetlinkd.Jetlinkd()
    d.failures = 6
    d.next_provision = time.monotonic() + jetlinkd.RETRY_BACKOFF_MAX
    provision = mock.Mock(return_value=True)
    self._step(d, provision)
    assert provision.call_count == 1
    assert d.failures == 0
    assert d.ready is True


class TestGadgetOnEnable(unittest.TestCase):
  """Boot sets the gadget up only with the link already on. A link turned on
  after boot has to get its gadget from the daemon, or the toggle does nothing
  until the next reboot."""

  def setUp(self):
    for target, new in (('accelerators', mock.Mock()),):
      p = mock.patch.object(jetlinkd, target, new)
      self.addCleanup(p.stop)
      p.start()
    for name, value in (('enabled', True), ('pending_shutdown', None), ('link_endpoint', None),
                        ('can_setup_gadget', True)):
      p = mock.patch.object(jetlinkd.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()
    self.setup = mock.patch.object(jetlinkd.helpers, 'setup_gadget', return_value=True)
    self.addCleanup(self.setup.stop)
    self.setup.start()

  def daemon(self):
    d = jetlinkd.Jetlinkd()
    d.warp_built = True
    for name in ('open_link', 'tune_vm'):
      p = mock.patch.object(d, name, mock.Mock(return_value=False))
      self.addCleanup(p.stop)
      p.start()
    return d

  def test_a_missing_gadget_is_created_before_the_link_is_opened(self):
    d = self.daemon()
    with mock.patch.object(jetlinkd.helpers, 'link_configured', return_value=False):
      d.step()
    assert jetlinkd.helpers.setup_gadget.call_count == 1
    assert d.open_link.call_count == 1

  def test_an_existing_gadget_is_left_alone(self):
    d = self.daemon()
    with mock.patch.object(jetlinkd.helpers, 'link_configured', return_value=True):
      d.step()
    assert jetlinkd.helpers.setup_gadget.call_count == 0
    assert d.open_link.call_count == 1

  def test_a_failed_setup_is_not_retried_every_tick(self):
    # the reason is in the status file for the offroad alert; a bash script per
    # tick would say it 120 times a minute
    d = self.daemon()
    jetlinkd.helpers.setup_gadget.return_value = False
    with mock.patch.object(jetlinkd.helpers, 'link_configured', return_value=False):
      for _ in range(3):
        d.step()
    assert jetlinkd.helpers.setup_gadget.call_count == 1
    assert d.open_link.call_count == 0
    assert d.next_gadget_attempt > time.monotonic()

  def test_a_device_that_cannot_make_one_still_tries_the_link(self):
    # a PC, or a build without the script: open_link fails and says why
    d = self.daemon()
    with mock.patch.object(jetlinkd.helpers, 'link_configured', return_value=False), \
         mock.patch.object(jetlinkd.helpers, 'can_setup_gadget', return_value=False):
      d.step()
    assert jetlinkd.helpers.setup_gadget.call_count == 0
    assert d.open_link.call_count == 1

  def test_tcp_needs_no_gadget(self):
    d = self.daemon()
    with mock.patch.object(jetlinkd.helpers, 'link_configured', return_value=False), \
         mock.patch.object(jetlinkd.helpers, 'link_endpoint', return_value=('10.0.0.2', 5599)):
      d.step()
    assert jetlinkd.helpers.setup_gadget.call_count == 0
    assert d.open_link.call_count == 1


class TestParked(unittest.TestCase):
  """Releasing the gadget once there is nothing to do, and taking it back."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    self.cache = FakeSpecCache()
    self.cache.spec = FakeSpec()
    self.model = self.tmp / 'big_driving_supercombo.onnx'
    self.model.write_bytes(b'x' * 4096)
    st = self.model.stat()
    self.cache.src = (str(self.model), st.st_mtime_ns, st.st_size)
    for target, new in (('spec_cache', self.cache), ('accelerators', mock.Mock())):
      p = mock.patch.object(jetlinkd, target, new)
      self.addCleanup(p.stop)
      p.start()
    for name, value in (('DORMANT', self.tmp / 'dormant'), ('SHUTDOWN_REQUEST', self.tmp / 'shutdown')):
      p = mock.patch.object(jetlinkd.helpers, name, value)
      self.addCleanup(p.stop)
      p.start()
    for name, value in (('enabled', True), ('host_attached', True), ('engine_ready_for', True),
                        ('selected_model', {'oid': self.cache.spec.sha256}),
                        ('shipped_model_path', self.model)):
      p = mock.patch.object(jetlinkd.helpers, name, return_value=value)
      self.addCleanup(p.stop)
      p.start()

  def daemon(self, ready=True):
    d = jetlinkd.Jetlinkd()
    d.client = mock.Mock()
    d.warp_built = True
    for name in ('open_link', 'close_link'):
      p = mock.patch.object(d, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(d, 'provision', mock.Mock(return_value=ready))
    self.addCleanup(p.stop)
    p.start()
    return d

  def test_holds_the_gadget_until_the_hold_has_passed(self):
    d = self.daemon()
    d.step()
    assert not d.dormant
    assert d.close_link.call_count == 0

  def test_releases_the_gadget_once_ready_and_parked_long_enough(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant
    assert d.close_link.call_count == 1
    assert jetlinkd.helpers.dormant()
    # and stays off the link while there is nothing to do
    d.step()
    assert d.open_link.call_count == 1  # only the first step presented it

  def test_not_ready_means_not_dormant(self):
    d = self.daemon(ready=False)
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert not d.dormant

  def test_a_cleared_readiness_wakes_it(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant
    d.started = time.monotonic()  # so the re-provision below does not put it straight back
    with mock.patch.object(jetlinkd.helpers, 'engine_ready_for', return_value=False):
      d.step()
    assert not d.dormant
    assert not jetlinkd.helpers.dormant()
    assert d.open_link.call_count == 2  # presented it again
    assert d.provision.call_count == 2  # and asked the server again

  def test_a_changed_model_wakes_it(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    d.started = time.monotonic()
    with mock.patch.object(jetlinkd.helpers, 'selected_model', return_value={'oid': 'other'}), \
         mock.patch.object(jetlinkd.helpers, 'shipped_model_path', return_value=None):
      d.step()
    assert not d.dormant

  def test_waking_for_work_that_is_done_goes_straight_back(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    with mock.patch.object(jetlinkd.helpers, 'engine_ready_for', return_value=False):
      d.step()
    assert d.dormant
    assert d.open_link.call_count == 2
    assert d.close_link.call_count == 2

  def test_nothing_selected_is_not_work(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    with mock.patch.object(jetlinkd.helpers, 'shipped_model_path', return_value=None):
      d.step()
    assert d.dormant

  def test_disabling_clears_the_marker(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    # Disabling drops JetlinkEngineReady. Mocked so this can never reach a
    # live params directory, conftest or no conftest.
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=False), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      d.step()
    assert not d.dormant
    assert not jetlinkd.helpers.dormant()

  def test_a_server_that_never_sleeps_keeps_the_gadget(self):
    # The whole point of letting go is a Jetson that suspends. One on ignition
    # power does not, and releasing left a powered, awake box unenumerated for
    # the entire parked period.
    d = self.daemon()
    d.server_sleeps = False
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert not d.dormant
    assert d.close_link.call_count == 0
    assert not jetlinkd.helpers.dormant()

  def test_a_server_that_sleeps_still_gets_the_gadget_back(self):
    d = self.daemon()
    d.server_sleeps = True
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant

  def test_a_server_too_old_to_say_keeps_the_release_it_has_always_had(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    assert d.server_sleeps is None
    d.step()
    assert d.dormant

  def test_the_hello_is_what_decides(self):
    d = jetlinkd.Jetlinkd()
    for hello, expected in (({'sleep_after': 120.0}, True), ({'sleep_after': 0}, False),
                            ({}, None), ({'sleep_after': 'soon'}, None)):
      d.note_sleep_after(hello)
      assert d.server_sleeps is expected, hello
      assert d.should_go_dormant() is (expected is not False)

  def test_a_shutdown_request_is_carried_to_the_jetson(self):
    d = self.daemon()
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant
    jetlinkd.helpers.request_shutdown('car battery')
    d.step()
    assert not d.dormant
    assert d.open_link.call_count == 2  # presented it again to wake it
    d.client.shutdown.assert_called_once_with('car battery', timeout=5.0)
    assert jetlinkd.helpers.pending_shutdown() is None

  def test_a_shutdown_request_is_consumed_even_when_it_fails(self):
    d = self.daemon()
    d.client.shutdown.side_effect = RuntimeError('link died')
    jetlinkd.helpers.request_shutdown('car battery')
    d.step()
    assert jetlinkd.helpers.pending_shutdown() is None

  def test_a_shutdown_request_waits_for_the_jetson_to_wake(self):
    d = self.daemon()
    attached = iter([False, False, True])
    with mock.patch.object(jetlinkd.helpers, 'host_attached', side_effect=lambda: next(attached)):
      jetlinkd.helpers.request_shutdown('car battery')
      d.step()
    d.client.shutdown.assert_called_once()


class TestGadgetOwnership(TestParked):
  """The daemon holds ep0 and the UDC bind for its whole life.

  What changes hands is the right to read the endpoint files: FunctionFS keeps
  a queued read queued until something completes it, so two readers on one
  endpoint take each other's replies.
  """

  @staticmethod
  def lend(d) -> None:
    d.lender = mock.Mock(lent=True)

  def test_a_borrower_keeps_the_gadget_on_the_bus_and_the_daemon_off_it(self):
    d = self.daemon()
    d.client.lendable = True
    self.lend(d)
    d.step()
    assert d.provision.call_count == 0, 'talked to the server over the borrower'
    assert d.open_link.call_count == 1, 'the gadget must stay bound for the drive'
    assert d.close_link.call_count == 0
    assert not d.dormant

  def test_a_borrower_wakes_a_dormant_daemon(self):
    d = self.daemon()
    d.server_sleeps = True
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant
    d.client.lendable = True
    self.lend(d)
    d.step()
    assert not d.dormant and not jetlinkd.helpers.dormant()

  def test_a_finished_provision_puts_the_endpoints_down(self):
    # the one re-enumeration this design still costs, spent parked rather than
    # at every ignition edge
    d = self.daemon()
    d.client.lendable = False
    d.step()
    assert d.close_link.call_count == 1
    assert d.open_link.call_count == 2, 'the gadget was left off the bus'

  def test_a_gadget_with_nothing_open_on_it_is_left_alone(self):
    d = self.daemon()
    d.client.lendable = True
    d.step()
    assert d.close_link.call_count == 0

  def test_going_dormant_does_not_bounce_first(self):
    d = self.daemon()
    d.client.lendable = False
    d.server_sleeps = True
    d.started = time.monotonic() - jetlinkd.DORMANT_HOLD
    d.step()
    assert d.dormant
    assert d.open_link.call_count == 1, 'presented the gadget on its way to letting go'

  def test_a_stuck_write_is_freed_by_the_owner(self):
    d = self.daemon()
    d.client.rebind.return_value = True
    assert d.bounce_gadget() is True
    d.client.rebind.assert_called_once()

  def test_nothing_to_bounce_is_not_an_error(self):
    d = self.daemon()
    d.client = None
    assert d.bounce_gadget() is False
    assert d.lendable() is False

  def test_a_shutdown_request_waits_for_the_borrower(self):
    d = self.daemon()
    d.client.lendable = True
    self.lend(d)
    jetlinkd.helpers.request_shutdown('car battery')
    with mock.patch.object(d, 'shutdown_jetson') as shutdown:
      d.step()
    shutdown.assert_not_called()
    assert jetlinkd.helpers.pending_shutdown() == 'car battery', 'the request was eaten'


class TestTimedOut(unittest.TestCase):
  def test_without_the_package_it_assumes_the_worst(self):
    # No jetlink installed means no way to tell a timeout from a desync, and
    # reopening a healthy link is cheaper than reusing a broken one.
    assert jetlinkd._timed_out(RuntimeError('boom')) is False

  def test_it_follows_jetlink_own_distinction(self):
    base = types.ModuleType('jetlink.transport.base')

    class LinkError(IOError):
      pass

    class LinkTimeout(LinkError):
      pass

    base.LinkError, base.LinkTimeout = LinkError, LinkTimeout
    mods = {'jetlink': types.ModuleType('jetlink'),
            'jetlink.transport': types.ModuleType('jetlink.transport'),
            'jetlink.transport.base': base}
    with mock.patch.dict(sys.modules, mods):
      assert jetlinkd._timed_out(LinkTimeout('no reply in time')) is True
      assert jetlinkd._timed_out(LinkError('stream desynced')) is False



class TestWarpFallback(TestParked):
  """scons builds the warp; build_warp only covers one that is missing."""

  def warp_daemon(self):
    d = self.daemon()
    d.warp_built = False
    jetlinkd.accelerators.report_progress.reset_mock()
    return d

  def test_a_warp_the_build_made_is_left_alone(self):
    # Reporting before checking put a "compiling the camera warp" through the
    # UI on every start for a warp that was already on disk.
    d = self.warp_daemon()
    with mock.patch.object(jetlinkd.warp_cache, 'is_cached', return_value=True), \
         mock.patch.object(jetlinkd.warp_cache, 'ensure') as ensure:
      d.build_warp()
    ensure.assert_not_called()
    assert jetlinkd.accelerators.report_progress.call_count == 0
    assert d.warp_thread is None

  def test_a_missing_warp_is_still_built(self):
    d = self.warp_daemon()
    with mock.patch.object(jetlinkd.warp_cache, 'is_cached', return_value=False), \
         mock.patch.object(jetlinkd.warp_cache, 'ensure', return_value=True) as ensure:
      d.build_warp()
      assert d.warp_thread is not None
      d.warp_thread.join(30)
    assert not d.warp_thread.is_alive()
    ensure.assert_called_once()
    assert jetlinkd.accelerators.report_progress.call_args.args[0] == 'warp'

  def test_it_is_attempted_once_per_run(self):
    d = self.warp_daemon()
    with mock.patch.object(jetlinkd.warp_cache, 'is_cached', return_value=True) as cached:
      d.build_warp()
      d.build_warp()
    cached.assert_called_once()


class TestVmTuning(unittest.TestCase):
  """A device with the link off runs stock values, one that turns it off gets
  them back, and a plain exit keeps them for the drive that follows."""

  # Stock AGNOS: ratio mode, so both *_bytes read 0 and the ratios carry the limit.
  STOCK = {'vm.dirty_bytes': '0', 'vm.dirty_background_bytes': '0', 'vm.min_free_kbytes': '7274',
           'vm.dirty_ratio': '20', 'vm.dirty_background_ratio': '5'}
  # What a restore of STOCK has to write: the kernel drops a 0 written to a
  # *_bytes key, and writing the ratio key is what zeroes it.
  RESTORED = ['vm.dirty_ratio=20', 'vm.dirty_background_ratio=5', 'vm.min_free_kbytes=7274']

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    proc = self.tmp / 'proc'
    for key, value in self.STOCK.items():
      path = proc / key.replace('.', '/')
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(value + '\n')
    self.record = self.tmp / 'prev'
    self.run_mock = mock.Mock()
    for p in (mock.patch.object(jetlinkd, 'PROC_SYS', proc),
              mock.patch.object(jetlinkd, 'SYSCTL_PREV', self.record),
              mock.patch.object(jetlinkd.subprocess, 'run', self.run_mock),
              mock.patch.object(jetlinkd.os, 'geteuid', return_value=1000),
              mock.patch.object(jetlinkd.helpers, 'DORMANT', self.tmp / 'dormant'),
              mock.patch.object(jetlinkd, 'accelerators', mock.Mock())):
      self.addCleanup(p.stop)
      p.start()

  def applied(self) -> list[str]:
    return [c.args[0][-1] for c in self.run_mock.call_args_list]

  def test_applied_on_start_and_kept_on_exit(self):
    d = jetlinkd.Jetlinkd()
    d.stop = True
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=True):
      d.run()
    ours = [f'{k}={v}' for k, v in jetlinkd.VM_SYSCTLS.items()]
    assert self.applied() == ours, "an exit is the ignition handoff; restoring here strips the drive of them"
    assert json.loads(self.record.read_text()) == self.STOCK, "the record is what a later disable restores to"

  def test_the_next_start_reapplies_without_touching_the_record(self):
    d = jetlinkd.Jetlinkd()
    d.stop = True
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=True):
      d.run()
    for key, value in jetlinkd.VM_SYSCTLS.items():
      (jetlinkd.PROC_SYS / key.replace('.', '/')).write_text(value)
    self.run_mock.reset_mock()
    d = jetlinkd.Jetlinkd()
    d.stop = True
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=True):
      d.run()
    assert self.applied() == [f'{k}={v}' for k, v in jetlinkd.VM_SYSCTLS.items()]
    assert json.loads(self.record.read_text()) == self.STOCK

  def test_the_record_is_written_before_anything_changes(self):
    with mock.patch.object(jetlinkd, '_write_sysctls') as write:
      jetlinkd.apply_vm_tuning()
    assert json.loads(self.record.read_text()) == self.STOCK
    write.assert_called_once_with(jetlinkd.VM_SYSCTLS)

  def test_an_existing_record_is_not_clobbered(self):
    # A previous run that was SIGKILLed left our values in /proc; reading them
    # now would record them as the stock ones and restore to them forever.
    self.record.write_text(json.dumps(self.STOCK))
    for key, value in jetlinkd.VM_SYSCTLS.items():
      (jetlinkd.PROC_SYS / key.replace('.', '/')).write_text(value)
    jetlinkd.apply_vm_tuning()
    assert json.loads(self.record.read_text()) == self.STOCK
    jetlinkd.restore_vm_tuning()
    assert self.applied()[-len(self.RESTORED):] == self.RESTORED

  def test_nothing_happens_when_disabled(self):
    d = jetlinkd.Jetlinkd()
    d.stop = True
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=False):
      d.run()
    assert self.run_mock.call_count == 0
    assert not self.record.exists()

  def test_disabling_mid_run_restores(self):
    d = jetlinkd.Jetlinkd()
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=True), \
         mock.patch.object(d, 'build_warp'), \
         mock.patch.object(d, 'open_link', return_value=False):
      d.step()
    assert d.vm_tuned
    with mock.patch.object(jetlinkd.helpers, 'enabled', return_value=False), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      d.step()
    assert not d.vm_tuned
    assert not self.record.exists()
    assert self.applied()[-1] == 'vm.min_free_kbytes=' + self.STOCK['vm.min_free_kbytes']

  def test_the_ratios_are_captured_in_the_record(self):
    jetlinkd.apply_vm_tuning()
    record = json.loads(self.record.read_text())
    assert record['vm.dirty_ratio'] == '20'
    assert record['vm.dirty_background_ratio'] == '5'

  def test_ratio_mode_is_restored_through_the_ratio_keys(self):
    # Measured on the comma: after our apply, `sysctl -w vm.dirty_bytes=0` and
    # a direct /proc write both return 0 and leave 16777216 in place.
    self.record.write_text(json.dumps(self.STOCK))
    jetlinkd.restore_vm_tuning()
    assert self.applied() == self.RESTORED
    assert not any(a.endswith('_bytes=0') for a in self.applied())

  def test_bytes_mode_is_restored_directly(self):
    prev = dict(self.STOCK, **{'vm.dirty_bytes': '33554432', 'vm.dirty_background_bytes': '4194304'})
    self.record.write_text(json.dumps(prev))
    jetlinkd.restore_vm_tuning()
    assert self.applied() == ['vm.dirty_bytes=33554432', 'vm.dirty_background_bytes=4194304',
                              'vm.min_free_kbytes=7274']

  def test_a_root_run_writes_proc_directly(self):
    with mock.patch.object(jetlinkd.os, 'geteuid', return_value=0):
      jetlinkd.apply_vm_tuning()
    assert self.run_mock.call_count == 0
    assert (jetlinkd.PROC_SYS / 'vm/min_free_kbytes').read_text() == '131072'


class BuildEtaTest(unittest.TestCase):
  """The estimate is what tells a driver watching "build 12%" whether that is
  five minutes or thirty."""

  def report(self, *args, size=1_850_000_000):
    seen = []
    with mock.patch.object(provision.helpers, 'selected_model', return_value={'size': size}), \
         mock.patch.object(provision.accelerators, 'report_progress', lambda *a: seen.append(a)):
      for call in args:
        provision._last_report = 0.0   # the 4 Hz throttle is not what is under test
        provision.report_with_eta(*call)
    return seen

  def test_the_build_stage_gets_a_time_remaining(self):
    seen = self.report(('build', 0.0, 'building the engine'), ('build', 0.8, 'building the engine'))
    self.assertEqual(seen[0][2], "about 5 min left")
    self.assertEqual(seen[1][2], "about 60s left")

  def test_other_stages_keep_their_own_message(self):
    # The upload already counts MB of MB, and a connect has nothing to predict.
    self.assertEqual(self.report(('upload', 0.5, '380/766 MB'))[0][2], '380/766 MB')

  def test_the_estimate_follows_the_measurements(self):
    self.assertTrue(100 <= provision.estimated_build_seconds(766_000_000) <= 200)     # built in 102 to 166 s
    self.assertTrue(230 <= provision.estimated_build_seconds(1_757_000_000) <= 320)   # 230 to 294 s

  def test_a_model_not_resolved_yet_is_survived(self):
    seen = self.report(('build', 0.3, 'building the engine'), size=None)
    self.assertEqual(seen[0][2], 'building the engine')

  def test_a_gigabyte_of_upload_is_not_a_gigabyte_of_param_writes(self):
    # 4 MB a chunk is 440 reports for a 1.7 GB model, and onroad that is IO the
    # recording would have to share the disk with.
    seen = []
    provision._last_report = 0.0
    with mock.patch.object(provision.accelerators, 'report_progress', lambda *a: seen.append(a)):
      for i in range(50):
        provision.report_with_eta('upload', i / 50, f'{i} MB')
      provision.report_with_eta('upload', 1.0, 'done')
    self.assertEqual(len(seen), 2, seen)


if __name__ == '__main__':
  unittest.main()
