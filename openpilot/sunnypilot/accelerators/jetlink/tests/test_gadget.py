"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget

# what the gadget owner must never end up importing. swaglog pulls all three in
# to publish a log line and costs 28 MB; params imports swaglog. Measured on the
# comma: this module plus the transport is 10.4 MB, against 47.5 MB for the
# daemon that imported the world
HEAVY = ('numpy', 'capnp', 'zmq', 'cereal')


class TestNothingHeavyIsReachable(unittest.TestCase):
  """The owner is only small while this holds, so it is a test and not a note."""

  def imported_by(self, module: str) -> set[str]:
    """Top-level packages a fresh interpreter has after importing `module`."""
    roots = 'sorted({m.split(".")[0] for m in sys.modules})'
    code = f'import sys, json; __import__("{module}"); print(json.dumps({roots}))'
    root = Path(__file__).resolve().parents[5]
    # this runner's own path, so the jetlink package is found wherever it is
    # checked out; a bare PYTHONPATH found the tree but not the submodule
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root), *sys.path])}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=env, cwd=str(root), timeout=120)
    self.assertEqual(out.returncode, 0, out.stderr)
    import json
    return set(json.loads(out.stdout))

  def test_the_gadget_core_stays_out_of_the_heavy_half(self):
    found = self.imported_by('openpilot.sunnypilot.accelerators.jetlink.gadget')
    self.assertEqual(sorted(found & set(HEAVY)), [],
                     'the gadget owner has to stay small; see the module docstring')

  def test_the_transport_the_owner_opens_is_light_too(self):
    try:
      import jetlink  # noqa: F401
    except ImportError:
      self.skipTest('the jetlink package is not checked out')
    found = self.imported_by('jetlink.transport.ffs')
    self.assertEqual(sorted(found & set(HEAVY)), [])


class TestParamsOffTheFilesystem(unittest.TestCase):
  """params.cc writes a value to a temp file, fsyncs it, renames it over the key
  and fsyncs the directory, so a plain read never sees a torn value."""

  def setUp(self):
    self.tmp = Path(tempfile.mkdtemp())
    (self.tmp / 'd').mkdir()
    self.enterContext(unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': str(self.tmp)}))
    os.environ.pop('OPENPILOT_PREFIX', None)

  def write(self, key: str, value: bytes) -> None:
    (self.tmp / 'd' / key).write_bytes(value)

  def test_the_path_follows_the_prefix(self):
    self.assertEqual(gadget.params_dir(), self.tmp / 'd')
    with unittest.mock.patch.dict(os.environ, {'OPENPILOT_PREFIX': 'abc123'}):
      self.assertEqual(gadget.params_dir(), self.tmp / 'abc123')

  def test_a_missing_param_is_not_a_false(self):
    # None and False are different answers: offroad treats an unwritten param
    # as parked, and enabled treats it as off
    self.assertIsNone(gadget.param_bool(gadget.P_ENABLED))
    self.assertFalse(gadget.enabled())
    self.assertTrue(gadget.offroad())

  def test_the_toggle_is_true_and_nothing_else(self):
    for raw, expected in ((b'1', True), (b'0', False), (b'', False), (b'true', True)):
      self.write(gadget.P_ENABLED, raw)
      self.assertIs(gadget.enabled(), expected, raw)

  def test_an_unreadable_store_is_not_an_error(self):
    with unittest.mock.patch.dict(os.environ, {'PARAMS_ROOT': '/nonexistent'}):
      self.assertIsNone(gadget.raw_param(gadget.P_ENABLED))
      self.assertFalse(gadget.enabled())


if __name__ == '__main__':
  unittest.main()
