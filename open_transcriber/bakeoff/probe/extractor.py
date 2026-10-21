"""The frozen-encoder ``Extractor`` protocol and a deterministic fake for CPU tests.

Every encoder in the bake-off is wrapped as an ``Extractor`` that declares its layers, native frame rate and
feature dims, and returns early/middle/final layer features for an audio file. Real extractors load heavy
weights and run on the GPU; they are implemented in the GPU window. All CPU-testable downstream code (the
probe body, heads, training loop, decode, decision) depends only on this protocol, so it runs against the
``FakeExtractor`` with no model download.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .. import paths

#: The three layer positions every encoder declares (a learned 3-way mix combines them).
LAYER_NAMES = ("early", "middle", "final")

#: Licence families allowed for training/shipping (model weights MIT; upstream MIT/Apache/BSD/CC0/CC BY).
_ALLOWED_FAMILIES = ("MIT", "Apache-2.0", "BSD", "CC0", "CC BY")


def _licence_allowed(licence: str) -> bool:
    """A licence is allowed only if it names an allowed family and carries no NC / ND restriction."""
    text = (licence or "").strip()
    upper = text.upper()
    if "-NC" in upper or " NC" in upper or "-ND" in upper:   # non-commercial / no-derivatives: rejected
        return False
    return any(text.startswith(a) for a in _ALLOWED_FAMILIES)


@dataclass(frozen=True)
class ExtractorDescriptor:
    """What an encoder is and how its features are shaped."""
    name: str
    code_licence: str
    weight_licence: str
    native_rate_hz: float          # the encoder's own frame rate (resampled to 50 Hz downstream)
    feature_dim: int               # feature width per layer
    layer_names: tuple[str, ...] = LAYER_NAMES
    device: str = "mps"            # where the encoder runs ("mps", "cpu")
    model_hash: str = ""           # a stable id for the weights, so the feature cache key changes with them
    note: str = ""                 # free-form note (e.g. "final-only", a drop reason, "auxiliary")
    dropped: bool = False          # a dropped candidate kept for the ledger; features() must raise

    @property
    def free_and_allowed(self) -> bool:
        return _licence_allowed(self.code_licence) and _licence_allowed(self.weight_licence)


@runtime_checkable
class Extractor(Protocol):
    def descriptor(self) -> ExtractorDescriptor: ...

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        """Features per declared layer: {layer_name: array of shape (frames, feature_dim)} at native_rate_hz."""
        ...


class FakeExtractor:
    """A deterministic tiny extractor for tests: features are a seeded function of the file path and layer.

    No audio is read; the number of frames is derived from an optional ``frames`` argument so tests can make
    short sequences. Produces the declared layers/shapes so the whole downstream stack is CPU-testable.
    """

    def __init__(self, name: str = "fake", native_rate_hz: float = 25.0, feature_dim: int = 8,
                 frames: int = 20):
        self._desc = ExtractorDescriptor(
            name=name, code_licence="MIT", weight_licence="MIT",
            native_rate_hz=native_rate_hz, feature_dim=feature_dim,
        )
        self._frames = frames

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        out = {}
        for i, layer in enumerate(self._desc.layer_names):
            seed = abs(hash((str(wav_path), layer))) % (2**32)
            rng = np.random.default_rng(seed)
            arr = rng.standard_normal((self._frames, self._desc.feature_dim)).astype(np.float32)
            # a tiny per-layer offset so the three layers are distinguishable
            out[layer] = arr + float(i)
        return out


class DroppedExtractor:
    """A descriptor-only placeholder for a candidate dropped by the prereg rule (e.g. HeartCodec).

    It carries the drop reason for the run ledger but refuses to extract: ``features()`` always raises, so a
    dropped encoder can never silently contribute features.
    """

    def __init__(self, descriptor: ExtractorDescriptor):
        self._desc = descriptor

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        raise RuntimeError(f"{self._desc.name} is dropped ({self._desc.note or 'prereg rule'}); "
                           "it must not be used for extraction")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def features_dir(encoder: str, root: Path | None = None) -> Path:
    """Where cached per-clip features for ``encoder`` live: ``.cache/ear/features/<encoder>/`` (gitignored)."""
    base = Path(root) if root is not None else paths.copyist_root() / ".cache" / "ear" / "features"
    return base / encoder


class CachedExtractor:
    """Wrap any ``Extractor`` so each clip's features are computed once and cached to a ``.npz`` on disk.

    The cache key is ``sha256(descriptor.name | descriptor.model_hash | clip sha256)``, so a window never
    re-extracts, and changing the weights (``model_hash``) or the audio invalidates the key. A ``.npz`` holds
    one array per declared layer. Delegates ``descriptor()`` to the wrapped extractor.
    """

    def __init__(self, inner: "Extractor", cache_dir: Path | None = None):
        self.inner = inner
        self._desc = inner.descriptor()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else features_dir(self._desc.name)

    def descriptor(self) -> ExtractorDescriptor:
        return self._desc

    def _key(self, wav_path: Path) -> str:
        clip_hash = _sha256_file(wav_path)
        token = f"{self._desc.name}|{self._desc.model_hash}|{clip_hash}".encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    def cache_path(self, wav_path: Path) -> Path:
        return self.cache_dir / f"{self._key(wav_path)}.npz"

    def features(self, wav_path: Path) -> dict[str, np.ndarray]:
        wav_path = Path(wav_path)
        path = self.cache_path(wav_path)
        if path.exists():
            with np.load(path) as data:
                return {name: data[name] for name in self._desc.layer_names}
        feats = self.inner.features(wav_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" to a name without that suffix, so build the temp path explicitly and
        # pass a file handle to keep the exact name, then rename atomically.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez(handle, **{name: feats[name] for name in self._desc.layer_names})
        tmp.replace(path)
        return feats


def build_extractor(name: str, device: str | None = None, cache_dir: Path | None = None) -> "Extractor":
    """Construct the real extractor for ``name`` (lazy heavy imports), wrapped in ``CachedExtractor``.

    ``heartcodec`` returns the drop-stub (not cached: it never extracts). Unknown names raise.
    """
    key = name.lower()
    if key == "heartcodec":
        from .extractors.heartcodec import heartcodec_descriptor
        return DroppedExtractor(heartcodec_descriptor())
    if key == "musicfm":
        from .extractors.musicfm import MusicFMExtractor
        inner: "Extractor" = MusicFMExtractor(device=device)
    elif key in ("dasheng", "dasheng-base"):
        from .extractors.dasheng import DashengExtractor
        inner = DashengExtractor(device=device)
    elif key in ("whisper", "whisper-base-en"):
        from .extractors.whisper_enc import WhisperEncExtractor
        inner = WhisperEncExtractor()
    else:
        raise ValueError(f"unknown extractor {name!r}")
    return CachedExtractor(inner, cache_dir=cache_dir)
