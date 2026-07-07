"""Self-built (non-diffusers) modeling for FSDP training.

Onboarding a model family:

- **Diffusers checkpoint** (has ``model_index.json``): nothing to do here.
  ``DiffusersModelBackend`` loads it; write only a ``configs/<family>.py``.

- **Native modeling** (official repo code, non-diffusers checkpoint): add
  ``models/<family>.py`` exposing the diffusers interface protocol the
  trainer relies on — ``from_pretrained(...)`` classmethod, a
  ``_no_split_modules`` list (consumed by FSDP wrapping) and
  ``enable_gradient_checkpointing()`` — plus a thin ``ModelBackend`` doing
  the loading, and a ``configs/<family>.py`` whose ``model_backend_path``
  points at it. Model-specific forward semantics live on the family config
  (``compute_noise_pred`` override), not in the trainer.
"""
