"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from tinygrad import Tensor, TinyJit


def prepare_reset(model):
  """Capture reset before driving, keeping the small JIT's buffer identities.

  Its history is stale after the Jetson ran, so fallback starts from the same
  zero history as modeld startup. Nothing is allocated or compiled on the
  failure frame.

  The small model is whatever bundle the user picked, stock modeld's or a
  modeld_v2 one, so the history is whatever GPU queues it has; the NPY tensors
  are zeroed through their numpy views.
  """
  queues = tuple(q for q in model.input_queues.values() if q.device != 'NPY')
  npy = model.numpy_inputs if hasattr(model, 'numpy_inputs') else model.npy

  @TinyJit
  def clear():
    Tensor.realize(*(q.assign(0) for q in queues))

  for _ in range(3):
    clear()

  def reset():
    clear()
    model.prev_desire.fill(0)
    for array in npy.values():
      array.fill(0)

  return reset
