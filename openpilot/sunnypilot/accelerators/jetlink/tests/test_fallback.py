"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np
from tinygrad import Tensor

from openpilot.sunnypilot.accelerators.jetlink.fallback import prepare_reset


def test_reset_clears_history_in_place_on_repeated_fallbacks():
  model = SimpleNamespace(
    input_queues={k: Tensor.zeros(4, 8, device='CPU').contiguous().realize()
                  for k in ('img_q', 'big_img_q', 'feat_q', 'desire_q')},
    prev_desire=np.ones(8), npy={'prev_feat': np.ones(32), 'desire': np.ones(8)})
  identities = {k: id(v) for k, v in model.input_queues.items()}
  reset = prepare_reset(model)
  for _ in range(3):
    for q in model.input_queues.values():
      q.assign(7).realize()
    model.prev_desire.fill(1)
    for v in model.npy.values():
      v.fill(1)
    reset()
    assert {k: id(v) for k, v in model.input_queues.items()} == identities
    for q in model.input_queues.values():
      np.testing.assert_array_equal(q.numpy(), 0)
    np.testing.assert_array_equal(model.prev_desire, 0)
    for v in model.npy.values():
      np.testing.assert_array_equal(v, 0)


def test_reset_takes_a_modeld_v2_bundle_as_it_is():
  # a split bundle: no feature queue, numpy inputs under numpy_inputs, and the
  # packed NPY tensor and the two transforms the warp reads are not queues
  packed = np.ones(16, dtype=np.float32)
  model = SimpleNamespace(
    input_queues={**{k: Tensor.zeros(4, 8, device='CPU').contiguous().realize() for k in ('img_q', 'big_img_q', 'desire_q')},
                  'packed_npy_inputs': Tensor(packed, device='NPY').realize()},
    prev_desire=np.ones(8),
    numpy_inputs={'desire': packed[:8], 'lateral_control_params': packed[8:10], 'prev_desired_curv': packed[10:],
                  'tfm': np.ones((3, 3), dtype=np.float32), 'big_tfm': np.ones((3, 3), dtype=np.float32)})
  reset = prepare_reset(model)
  for q in ('img_q', 'big_img_q', 'desire_q'):
    model.input_queues[q].assign(7).realize()
  reset()
  for q in ('img_q', 'big_img_q', 'desire_q'):
    np.testing.assert_array_equal(model.input_queues[q].numpy(), 0)
  np.testing.assert_array_equal(model.prev_desire, 0)
  np.testing.assert_array_equal(packed, 0)  # every numpy input is a view into it
  np.testing.assert_array_equal(model.input_queues['packed_npy_inputs'].numpy(), 0)
