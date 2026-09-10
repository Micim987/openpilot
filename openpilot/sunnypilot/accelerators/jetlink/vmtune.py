"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The VM tuning the gadget needs, and how to put it back.

Standard library only: this belongs to the process that owns the gadget, which
has to stay small (see gadget.py).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from openpilot.sunnypilot.accelerators.jetlink import gadget

# loggerd's dirty pages pile up until the kernel reclaims them synchronously,
# right while a FunctionFS transfer allocates its buffer: gadget reads stalled
# 200-350 ms, past backend.INFERENCE_TIMEOUT, and the big model fell back.
# Capping dirty memory and holding a free-memory floor took the worst frame
# from 244 to 72 ms with no lagging frames over 20 min.
#
# System-wide, since the gadget read shares the kernel with every writer.
# Applied here so a device with the link off runs stock values, which are
# recorded in SYSCTL_PREV and put back only on disable. Never restored on
# exit: manager stops this daemon at ignition, exactly when the contention
# starts, so modeld would get stock values every drive. A reboot resets them
VM_SYSCTLS = {
  'vm.dirty_bytes': '16777216',
  'vm.dirty_background_bytes': '8388608',
  'vm.min_free_kbytes': '131072',
}
# stock AGNOS runs the dirty limits in ratio mode, so both *_bytes keys read 0,
# and the kernel silently drops a 0 written back to them. Writing the ratio key
# is what zeroes the bytes key, so the ratios are recorded alongside
VM_RATIO_KEYS = {
  'vm.dirty_bytes': 'vm.dirty_ratio',
  'vm.dirty_background_bytes': 'vm.dirty_background_ratio',
}
SYSCTL_PREV = Path('/dev/shm/jetlink-sysctl-prev')
PROC_SYS = Path('/proc/sys')


def _read_sysctls(keys) -> dict[str, str]:
  values = {}
  for key in keys:
    try:
      values[key] = (PROC_SYS / key.replace('.', '/')).read_text().strip()
    except OSError:
      gadget.log.exception(f"jetlink: could not read {key}")
  return values


def _write_sysctls(values: dict[str, str]) -> None:
  # sudo -n sysctl is the same privilege setup_gadget.sh uses; root writes /proc.
  # One key per call, so a value the kernel rejects does not take the rest with it
  for key, value in values.items():
    try:
      if os.geteuid() == 0:
        (PROC_SYS / key.replace('.', '/')).write_text(value)
      else:
        subprocess.run(['sudo', '-n', 'sysctl', '-w', f'{key}={value}'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
      gadget.log.exception(f"jetlink: could not set {key}={value}")


def apply_vm_tuning() -> None:
  """Record the stock values once, then apply ours."""
  if not SYSCTL_PREV.exists():
    prev = _read_sysctls([*VM_SYSCTLS, *VM_RATIO_KEYS.values()])
    if prev:
      try:
        SYSCTL_PREV.write_text(json.dumps(prev))
      except OSError:
        gadget.log.exception("jetlink: could not record the previous sysctls")
  _write_sysctls(VM_SYSCTLS)


def restore_vm_tuning() -> None:
  """Put the recorded values back and drop the record."""
  try:
    prev = json.loads(SYSCTL_PREV.read_text())
  except FileNotFoundError:
    return
  except (OSError, ValueError):
    gadget.log.exception("jetlink: unreadable sysctl record, leaving the values as they are")
    prev = {}
  if isinstance(prev, dict):
    values = {}
    for key in VM_SYSCTLS:
      if key not in prev:
        continue
      ratio = VM_RATIO_KEYS.get(key)
      if str(prev[key]) == '0' and ratio in prev:
        # the kernel drops a 0 written to a *_bytes key; the ratio key is the
        # way back to ratio mode
        values[ratio] = str(prev[ratio])
      else:
        values[key] = str(prev[key])
    _write_sysctls(values)
  try:
    SYSCTL_PREV.unlink()
  except OSError:
    pass
