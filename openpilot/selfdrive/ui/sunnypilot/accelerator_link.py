"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The user's say over the accelerator link, shared by the mici and the tici
models panels. On is the only enable: an absent param is off, the backend
never decides on its own. The backend reads the param, the panels only write
it, so the name lives here and never on screen.
"""
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot import accelerators

LINK_PARAM = "JetlinkEnabled"


def link_enabled() -> bool:
  """Never raises: a params library older than the key would otherwise take the
  settings panel down, the guard accelerators.progress() has."""
  try:
    return bool(ui_state.params.get(LINK_PARAM))
  except Exception:
    return False


def set_link_enabled(enabled: bool) -> None:
  try:
    ui_state.params.put_bool(LINK_PARAM, enabled, block=True)
    # manager caches which modeld it runs; the link decides that
    ui_state.params.remove('ModelRunnerTypeCache')
  except Exception:
    pass  # the same unknown-key case as the read; nothing the panel can do about it


def link_toggle_meaningful() -> bool:
  """Whether to show the toggle at all. A plain device with no accelerator, no
  complaint and the link off must not. ready() is on the list for the link whose
  engine is cached while the hardware is out of the car: that is exactly when
  someone wants to turn it off."""
  return (accelerators.present() or accelerators.ready() or link_enabled()
          or accelerators.unavailable_reason() is not None)


def selected_accelerator_model() -> str:
  """The picked accelerator model's name, '' when the link has no registry."""
  return next((m['name'] for m in accelerators.model_choices() if m['selected']), '')
