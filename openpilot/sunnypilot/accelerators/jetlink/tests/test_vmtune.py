"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The VM tuning the gadget needs, and how it is put back.

These numbers were measured: capping dirty memory and holding a free-memory
floor took the worst FunctionFS frame from 244 ms to 72 ms over 20 minutes.
The awkward parts are the ones with a device behind them, so they are the ones
worth keeping honest: stock AGNOS runs the dirty limits in ratio mode, and the
kernel silently drops a 0 written back to a *_bytes key.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpilot.sunnypilot.accelerators.jetlink import gadget, owner, vmtune


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
    for p in (mock.patch.object(vmtune, 'PROC_SYS', proc),
              mock.patch.object(vmtune, 'SYSCTL_PREV', self.record),
              mock.patch.object(vmtune.subprocess, 'run', self.run_mock),
              mock.patch.object(vmtune.os, 'geteuid', return_value=1000),
              mock.patch.object(gadget, 'DORMANT', self.tmp / 'dormant'),
              mock.patch.object(gadget, 'link_endpoint', mock.Mock(return_value=None))):
      self.addCleanup(p.stop)
      p.start()

  def owner(self, enabled=True):
    """An owner with the gadget already presented and nothing else to do."""
    o = owner.Owner()
    o.lender = mock.Mock(lent=False, listening=True)
    o.transport = mock.Mock(lendable=True)
    o.seen = dict.fromkeys(owner.WATCHED, 0)
    o.had_host = True
    for name in ('open_link', 'spawn_worker', 'settle', 'go_dormant'):
      p = mock.patch.object(o, name, mock.Mock(return_value=True))
      self.addCleanup(p.stop)
      p.start()
    p = mock.patch.object(gadget, 'enabled', return_value=enabled)
    self.addCleanup(p.stop)
    p.start()
    return o

  def applied(self) -> list[str]:
    return [c.args[0][-1] for c in self.run_mock.call_args_list]

  def test_applied_on_start_and_kept_on_exit(self):
    o = self.owner()
    o.step()
    ours = [f'{k}={v}' for k, v in vmtune.VM_SYSCTLS.items()]
    assert self.applied() == ours, "an exit is the ignition handoff; restoring here strips the drive of them"
    assert json.loads(self.record.read_text()) == self.STOCK, "the record is what a later disable restores to"

  def test_the_next_start_reapplies_without_touching_the_record(self):
    self.owner().step()
    for key, value in vmtune.VM_SYSCTLS.items():
      (vmtune.PROC_SYS / key.replace('.', '/')).write_text(value)
    self.run_mock.reset_mock()
    self.owner().step()
    assert self.applied() == [f'{k}={v}' for k, v in vmtune.VM_SYSCTLS.items()]
    assert json.loads(self.record.read_text()) == self.STOCK

  def test_the_record_is_written_before_anything_changes(self):
    with mock.patch.object(vmtune, '_write_sysctls') as write:
      vmtune.apply_vm_tuning()
    assert json.loads(self.record.read_text()) == self.STOCK
    write.assert_called_once_with(vmtune.VM_SYSCTLS)

  def test_an_existing_record_is_not_clobbered(self):
    # A previous run that was SIGKILLed left our values in /proc; reading them
    # now would record them as the stock ones and restore to them forever.
    self.record.write_text(json.dumps(self.STOCK))
    for key, value in vmtune.VM_SYSCTLS.items():
      (vmtune.PROC_SYS / key.replace('.', '/')).write_text(value)
    vmtune.apply_vm_tuning()
    assert json.loads(self.record.read_text()) == self.STOCK
    vmtune.restore_vm_tuning()
    assert self.applied()[-len(self.RESTORED):] == self.RESTORED

  def test_nothing_happens_when_disabled(self):
    self.owner(enabled=False).step()
    assert self.run_mock.call_count == 0
    assert not self.record.exists()

  def test_disabling_mid_run_restores(self):
    o = self.owner()
    o.step()
    assert o.vm_tuned
    with mock.patch.object(gadget, 'enabled', return_value=False):
      o.step()
    assert not o.vm_tuned
    assert not self.record.exists()
    assert self.applied()[-1] == 'vm.min_free_kbytes=' + self.STOCK['vm.min_free_kbytes']

  def test_the_ratios_are_captured_in_the_record(self):
    vmtune.apply_vm_tuning()
    record = json.loads(self.record.read_text())
    assert record['vm.dirty_ratio'] == '20'
    assert record['vm.dirty_background_ratio'] == '5'

  def test_ratio_mode_is_restored_through_the_ratio_keys(self):
    # Measured on the comma: after our apply, `sysctl -w vm.dirty_bytes=0` and
    # a direct /proc write both return 0 and leave 16777216 in place.
    self.record.write_text(json.dumps(self.STOCK))
    vmtune.restore_vm_tuning()
    assert self.applied() == self.RESTORED
    assert not any(a.endswith('_bytes=0') for a in self.applied())

  def test_bytes_mode_is_restored_directly(self):
    prev = dict(self.STOCK, **{'vm.dirty_bytes': '33554432', 'vm.dirty_background_bytes': '4194304'})
    self.record.write_text(json.dumps(prev))
    vmtune.restore_vm_tuning()
    assert self.applied() == ['vm.dirty_bytes=33554432', 'vm.dirty_background_bytes=4194304',
                              'vm.min_free_kbytes=7274']

  def test_a_root_run_writes_proc_directly(self):
    with mock.patch.object(vmtune.os, 'geteuid', return_value=0):
      vmtune.apply_vm_tuning()
    assert self.run_mock.call_count == 0
    assert (vmtune.PROC_SYS / 'vm/min_free_kbytes').read_text() == '131072'
