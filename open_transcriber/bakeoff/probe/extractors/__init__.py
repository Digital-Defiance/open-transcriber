"""Real frozen-encoder extractors for the bake-off (loaded only in the GPU window / short probes).

Each module exposes one class satisfying the ``Extractor`` protocol and declaring its ``ExtractorDescriptor``
exactly as the 21 October smoke test documented (``.agents/tasks/ear-bakeoff-2026-10-20/encoder-smoke-2026-10-21.md``):

- ``musicfm``  MusicFM FMA checkpoint (MIT), 24 kHz mono, hidden 3/7/12, (frames,1024) @ 25 Hz, mps. Reference.
- ``dasheng``  Dasheng base (Apache), final (frames,768) @ 25 Hz, mps; early/middle via the real hidden-states
  API or final-only (never the misaligned ``.blocks`` hooks).
- ``whisper_enc``  Whisper base.en (MIT), encoder blocks 1/3/6, (1500,512) @ 50 Hz, CPU only. Auxiliary.
- ``heartcodec``  descriptor only; DROPPED (unsupported on MPS without a clear pilot lead).
- ``basic_pitch_side``  the Basic Pitch side input + encoder-free control, run as a ``.venv`` subprocess
  (TensorFlow never enters the ear env).

Heavy imports (torch, transformers, whisper, dasheng) are deliberately inside the methods so importing this
subpackage stays cheap and CPU-safe.
"""
