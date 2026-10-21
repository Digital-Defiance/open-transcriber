"""CREPE pitch tracking for chart QA: torchcrepe 0.0.24's "full" weights through ONNX Runtime, decoded in numpy.

This re-implements torchcrepe's inference path (MIT; Copyright (c) 2020 Max Morrison) without torch or librosa:

- audio: ``scipy.io.wavfile``, mono mean, ``scipy.signal.resample_poly`` to 16 kHz (torchcrepe uses resampy);
- framing: 1024-sample frames every 160 samples (10 ms), zero-padded by 512 at each end, each frame
  mean-centred and divided by its standard deviation (unbiased, as torch's ``std``) with a 1e-10 floor;
- the network: ``crepe-full.onnx``, exported once from the wheel's ``assets/full.pth``
  (``tools/export_crepe_onnx.py``), run on the CPU;
- decoding: bins outside [fmin, fmax] set to -inf, a softmax over bins, then Viterbi with torchcrepe's
  triangular transition (width 12) exactly as ``librosa.sequence.viterbi`` computes it (log with the dtype's
  tiny epsilon, uniform start, first maximum on ties). Periodicity is the network's output at the decoded bin.
- pitch: torchcrepe's bin-to-cents map, 20 cents a bin from 1997.379 cents, without its random dither
  (so results are deterministic).

The evaluation environment then needs no torch and no copyleft packages.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import paths

SAMPLE_RATE = 16000
WINDOW = 1024
HOP = 160
PITCH_BINS = 360
CENTS_PER_BIN = 20
CENTS_OFFSET = 1997.3794084376191
#: torchcrepe-0.0.24-py3-none-any.whl on PyPI.
WHEEL_SHA256 = "ec054c23c9d45328f213f93a0131570a3f0e5903e9382792bed95f17a8c36d5a"
#: crepe-full.onnx exported by tools/export_crepe_onnx.py (opset 17) from that wheel's full.pth.
MODEL_SHA256 = "26f8366666fd9831d92fb168a9652fc89cffa1b0b2af73fd5ed974b20f1939bf"
#: The export's parity check (2026-10-19, torch 2.8.0, onnxruntime 1.31.0): 2,000 random frames, max |diff|
#: 1.4e-6, argmax equal on all; a 220->440 Hz sweep, numpy pipeline vs torchcrepe's Viterbi bins equal on all
#: 301 frames, median error 5.9 cents.


def model_path() -> Path:
    return paths.data_dir() / "models" / "crepe-full.onnx"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A WAV file as float32 [samples, channels] in [-1, 1], and its rate."""
    from scipy.io import wavfile

    rate, data = wavfile.read(str(path))
    if data.ndim == 1:
        data = data[:, None]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    return audio, int(rate)


def to_16k_mono(audio: np.ndarray, rate: int) -> np.ndarray:
    """Mono mean, then polyphase resampling to 16 kHz (float32)."""
    from scipy.signal import resample_poly

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if rate != SAMPLE_RATE:
        ratio = Fraction(SAMPLE_RATE, rate)
        mono = resample_poly(mono.astype(np.float64), ratio.numerator, ratio.denominator)
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_16k(path: Path) -> np.ndarray:
    audio, rate = read_wav(path)
    return to_16k_mono(audio, rate)


def frame_count(samples: int, hop: int = HOP) -> int:
    return 1 + samples // hop


def raw_frames(audio: np.ndarray, start: int = 0, stop: int | None = None, hop: int = HOP) -> np.ndarray:
    """Frames ``start``..``stop`` (exclusive) of the zero-padded signal, before normalisation: [n, 1024]."""
    total = frame_count(len(audio), hop)
    stop = total if stop is None else min(stop, total)
    padded = np.pad(audio.astype(np.float32), (WINDOW // 2, WINDOW // 2))
    starts = hop * np.arange(start, stop)
    return padded[starts[:, None] + np.arange(WINDOW)[None, :]]


def normalise(frames: np.ndarray) -> np.ndarray:
    """torchcrepe's per-frame normalisation: subtract the mean, divide by max(1e-10, unbiased std)."""
    frames = frames - frames.mean(axis=1, keepdims=True, dtype=np.float32)
    scale = np.maximum(np.float32(1e-10), frames.std(axis=1, ddof=1, keepdims=True, dtype=np.float32))
    return (frames / scale).astype(np.float32)


class Model:
    """The ONNX network on the CPU (deterministic; 4 intra-op threads by default)."""

    def __init__(self, path: Path | None = None, threads: int = 4, check_hash: bool = True):
        import onnxruntime as ort

        path = Path(path or model_path())
        if check_hash:
            from .fetch import sha256_file

            if sha256_file(path) != MODEL_SHA256:
                raise ValueError(f"{path.name}: unexpected SHA-256; re-export it with tools/export_crepe_onnx.py")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def activations(self, frames: np.ndarray, batch: int = 1024) -> np.ndarray:
        """Sigmoid outputs [n, 360] for normalised frames [n, 1024]."""
        out = np.empty((len(frames), PITCH_BINS), dtype=np.float32)
        for i in range(0, len(frames), batch):
            out[i:i + batch] = self.session.run(None, {self.input: frames[i:i + batch]})[0]
        return out


def frequency_to_bin(frequency: float, ceil: bool = False) -> int:
    cents = 1200.0 * math.log2(frequency / 10.0)
    value = (cents - CENTS_OFFSET) / CENTS_PER_BIN
    return int(math.ceil(value) if ceil else math.floor(value))


def bins_to_cents(bins: np.ndarray) -> np.ndarray:
    return CENTS_PER_BIN * np.asarray(bins, dtype=np.float64) + CENTS_OFFSET


def bins_to_midi(bins: np.ndarray) -> np.ndarray:
    """MIDI pitch (float) of each bin: 10 Hz * 2^(cents/1200), then 69 + 12*log2(f/440)."""
    frequency = 10.0 * 2.0 ** (bins_to_cents(bins) / 1200.0)
    return 69.0 + 12.0 * np.log2(frequency / 440.0)


def _transition() -> np.ndarray:
    xx, yy = np.meshgrid(range(PITCH_BINS), range(PITCH_BINS))
    transition = np.maximum(12 - abs(xx - yy), 0)
    return transition / transition.sum(axis=1, keepdims=True)


TRANSITION = _transition()


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row softmax in float32; -inf entries become 0."""
    logits = logits.astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def viterbi(probs: np.ndarray, transition: np.ndarray = TRANSITION) -> np.ndarray:
    """Most likely state path for observation probabilities [n_steps, n_states], as librosa computes it."""
    probs = np.asarray(probs)
    epsilon = np.finfo(probs.dtype).tiny
    n_steps, n_states = probs.shape
    log_trans = np.log(transition + epsilon)
    log_prob = np.log(probs + epsilon)
    p_init = np.empty(n_states)
    p_init.fill(1.0 / n_states)
    log_p_init = np.log(p_init + epsilon)
    trans_t = np.ascontiguousarray(log_trans.T)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)
    value = log_prob[0] + log_p_init
    rows = np.arange(n_states)
    for t in range(1, n_steps):
        trans_out = value[None, :] + trans_t
        best = trans_out.argmax(axis=1)
        ptr[t] = best
        value = log_prob[t] + trans_out[rows, best]
    state = np.zeros(n_steps, dtype=np.int64)
    state[-1] = int(np.argmax(value))
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state


def decode(activations: np.ndarray, fmin: float = 50.0, fmax: float = 1100.0) -> tuple[np.ndarray, np.ndarray]:
    """(bins, periodicity) per frame, as torchcrepe's postprocess with the Viterbi decoder."""
    logits = activations.astype(np.float32).copy()
    logits[:, :frequency_to_bin(fmin)] = -np.inf
    logits[:, frequency_to_bin(fmax, ceil=True):] = -np.inf
    bins = viterbi(softmax(logits))
    periodicity = logits[np.arange(len(bins)), bins]
    return bins, periodicity.astype(np.float32)


def frame_rms(audio: np.ndarray, hop: int = HOP, chunk: int = 4096) -> np.ndarray:
    """RMS of each un-normalised analysis frame (the same 1024-sample frames CREPE sees)."""
    total = frame_count(len(audio), hop)
    out = np.empty(total, dtype=np.float32)
    for i in range(0, total, chunk):
        frames = raw_frames(audio, i, i + chunk, hop).astype(np.float64)
        out[i:i + len(frames)] = np.sqrt((frames ** 2).mean(axis=1))
    return out


def track(audio: np.ndarray, model: Model, fmin: float = 50.0, fmax: float = 1100.0,
          chunk: int = 4096) -> dict[str, np.ndarray]:
    """CREPE over a whole 16 kHz signal: bins, MIDI pitch, periodicity and frame RMS (10 ms frames)."""
    total = frame_count(len(audio))
    acts = np.empty((total, PITCH_BINS), dtype=np.float32)
    for i in range(0, total, chunk):
        acts[i:i + chunk] = model.activations(normalise(raw_frames(audio, i, i + chunk)))
    bins, periodicity = decode(acts, fmin, fmax)
    return {"bins": bins.astype(np.int16), "midi": bins_to_midi(bins).astype(np.float32),
            "periodicity": periodicity, "rms": frame_rms(audio)}


def save_track(path: Path, result: dict[str, np.ndarray], provenance: dict) -> None:
    """Write a track as ``.npz`` with its provenance (hashes, settings) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, **result, provenance=np.array(json.dumps(provenance, sort_keys=True)))
    tmp.replace(path)


def load_track(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        result = {k: data[k] for k in data.files if k != "provenance"}
        provenance = json.loads(str(data["provenance"]))
    return result, provenance
