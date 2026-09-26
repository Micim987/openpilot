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
import types
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import gadget, jetlinkd, provision


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

  def test_a_ready_param_is_still_checked_with_the_server(self):
    # the Jetson's cache can be pruned or re-flashed under a param that says
    # ready. A run provisions once and then exits, so the check is once a run
    # and there is no second call to skip
    self.cache.store(FakeSpec(), self.model)
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'engine_ready_for', return_value=True), \
         mock.patch.object(jetlinkd.helpers, 'set_engine_ready') as ready:
      assert d.provision() is True
    assert d.client.ensure_engine.call_count == 1
    assert self.hashed == [], 'read the model to answer a question the server answers'
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
    # a build started while parked can still be running when the driver pulls
    # away. The owner sends a stop at the onroad transition so modeld can take
    # the endpoints; the server's build thread carries on and modeld picks the
    # engine up over its own link
    d = jetlinkd.Jetlinkd()
    d.client = serving_client()
    with mock.patch.object(jetlinkd.helpers, 'set_engine_ready'):
      d.provision()
    should_stop = d.client.ensure_engine.call_args.kwargs['should_stop']
    assert should_stop() is False
    d.request_stop()
    assert should_stop() is True


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



class TestWarpFallback(unittest.TestCase):
  """scons builds the warp; build_warp only covers one that is missing."""

  def setUp(self):
    p = mock.patch.object(jetlinkd, 'accelerators', mock.Mock())
    self.addCleanup(p.stop)
    p.start()

  def warp_daemon(self):
    d = jetlinkd.Jetlinkd()
    d.warp_built = False
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


class TestTheRun(unittest.TestCase):
  """One round, then the process exits. What it leaves behind is what the owner
  cannot work out for itself."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    for target, new in (('accelerators', mock.Mock()), ('warp_cache', mock.Mock())):
      p = mock.patch.object(jetlinkd, target, new)
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(gadget, 'STATE', self.tmp / 'state')
    self.addCleanup(p.stop)
    p.start()

  def worker(self, work=True):
    d = jetlinkd.Jetlinkd()
    d.warp_built = True
    for name, value in (('has_work', work), ('open_link', True), ('provision', True)):
      p = mock.patch.object(d, name, mock.Mock(return_value=value))
      self.addCleanup(p.stop)
      p.start()
    for name in ('enabled', 'migrate_selection', 'pending_shutdown', 'wait_for_host'):
      p = mock.patch.object(jetlinkd.helpers, name,
                            mock.Mock(return_value={'enabled': True, 'wait_for_host': True}.get(name)))
      self.addCleanup(p.stop)
      p.start()
    return d

  def state(self) -> dict:
    return json.loads(gadget.STATE.read_text())

  def test_nothing_to_do_never_opens_the_link(self):
    d = self.worker(work=False)
    assert d.run() is True
    d.open_link.assert_not_called()
    assert self.state()['unfinished'] is False

  def test_a_finished_round_says_so_and_lets_the_link_go(self):
    d = self.worker()
    d.client = mock.Mock()
    assert d.run() is True
    d.provision.assert_called_once()
    assert self.state()['unfinished'] is False
    assert d.client is None, 'left the gadget open after the run'

  def test_a_round_that_fails_leaves_the_work_for_the_next_one(self):
    d = self.worker()
    d.provision.side_effect = RuntimeError('the jetson went away')
    assert d.run() is False
    assert self.state()['unfinished'] is True

  def test_no_jetson_is_left_for_the_next_run(self):
    d = self.worker()
    jetlinkd.helpers.wait_for_host.return_value = False
    assert d.run() is False
    d.provision.assert_not_called()
    assert self.state()['unfinished'] is True

  def test_what_the_far_end_does_when_the_gadget_goes_is_recorded(self):
    # the owner never speaks the protocol, so this is the only way it learns
    d = self.worker()
    d.note_sleep_after({'sleep_after': 0.0})
    d.run()
    assert self.state()['sleep_after'] == 0.0

  def test_a_shutdown_request_is_the_whole_round(self):
    d = self.worker()
    jetlinkd.helpers.pending_shutdown.return_value = 'car battery'
    with mock.patch.object(d, 'shutdown_jetson') as shutdown:
      assert d.run() is True
    shutdown.assert_called_once_with('car battery')
    d.provision.assert_not_called()

  def test_the_link_off_is_not_a_round(self):
    d = self.worker()
    jetlinkd.helpers.enabled.return_value = False
    assert d.run() is True
    d.open_link.assert_not_called()


class BuildEtaTest(unittest.TestCase):
  """The estimate is what tells a driver watching "build 12%" whether that is
  five minutes or thirty."""

  def report(self, *args, size=1_850_000_000):
    seen = []
    with mock.patch.object(provision.helpers, 'selected_model', return_value={'size': size}), \
         mock.patch.object(provision.accelerators, 'report_progress', lambda *a: seen.append(a)):
      for call in args:
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



if __name__ == '__main__':
  unittest.main()
