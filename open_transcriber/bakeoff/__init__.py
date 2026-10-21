"""The encoder ("ear") bake-off package.

Generic, reusable evaluation and job infrastructure ported from the stage-0a judge (source ``aa88f5a``),
plus the bake-off's own sampler-material, split, probe, training, evaluation and decision code.

Two boundaries this package keeps:

- **NC data stays test-only.** ``manifests``, ``refs``, ``rwc_midi``, ``ultrastar``, ``qa``, ``crepe``,
  ``onsets`` and ``lyrics_guard`` touch RWC/karaoke (CC BY-NC) material for evaluation/probe-eval only.
  Product and training code must not import them; ``tests/test_import_boundaries.py`` enforces that.
- **No SheetSage.** No SheetSage runner, environment or output format lives here; SheetSage is never a
  label, teacher or tuning signal, and this package does not depend on or resume any SheetSage run.
"""
