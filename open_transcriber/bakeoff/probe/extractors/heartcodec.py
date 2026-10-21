"""HeartCodec encoder — DROPPED per the prereg compute rule (descriptor only, never extracts).

The 21 October smoke test confirmed HeartCodec is licence-clean (Apache-2.0) but unsupported on MPS: its
internal 24k->16k torchaudio sinc resample uses ``conv1d(stride=24000)``, producing >65536 output channels,
a hard MPS limitation (``NotImplementedError``, not caught by ``PYTORCH_ENABLE_MPS_FALLBACK``). CPU works at
~5 s/clip (5-8x slower than the mps encoders). Under the prereg rule — "unsupported on MPS without a clear
pilot lead" — it is dropped. This module keeps only its descriptor so the run ledger records *why*; it is
never instantiated for extraction (``build_extractor("heartcodec")`` returns a ``DroppedExtractor``).
"""

from __future__ import annotations

from ..extractor import ExtractorDescriptor

DROP_REASON = ("dropped: unsupported on MPS without a clear pilot lead "
               "(24k->16k resample exceeds the 65536-channel conv1d limit)")


def heartcodec_descriptor() -> ExtractorDescriptor:
    """The descriptor-only drop entry for HeartCodec (licence-clean, but dropped)."""
    return ExtractorDescriptor(
        name="heartcodec", code_licence="Apache-2.0", weight_licence="Apache-2.0",
        native_rate_hz=12.5, feature_dim=512, device="cpu",
        model_hash="", note=DROP_REASON, dropped=True,
    )
