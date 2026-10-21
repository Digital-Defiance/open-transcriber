"""HeartCodec-encoder (Apache-2.0) load-and-extract smoke driver — standalone encoder, continuous features.

MPS support is the flagged risk (card is CUDA-focused). We load ONLY HeartCodecEncoder from the downloaded
encoder_config.json + encoder.safetensors (no decoder, no quantizer, no bitsandbytes/torchtune) and call
``forward(audio_24k)`` to get continuous pre-quantization features. There is one "layer" (the encoder output);
we report it under all three names so the table is comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from encoder_smoke import pick_clips, run_smoke  # noqa: E402

HEARTLIB_SRC = Path("/Volumes/Code/copyist/.cache/ear/encoders/heartlib/src")
sys.path.insert(0, str(HEARTLIB_SRC))
CKPT = Path("/Volumes/Code/copyist/.cache/ear/encoders/heartcodec-oss-encoder")
SR = 24000


def _import_encoder_cls():
    """Import HeartCodecEncoder without running heartlib/__init__ (which pulls in torchtune via the
    generation pipeline). We load the submodule files directly by path."""
    import importlib.util
    import types

    # register empty parent packages so relative imports inside the submodules resolve
    for pkg in ("heartlib", "heartlib.heartcodec", "heartlib.heartcodec.models"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(HEARTLIB_SRC / Path(*pkg.split(".")))]
            sys.modules[pkg] = mod
    enc_path = HEARTLIB_SRC / "heartlib" / "heartcodec" / "models" / "encoder.py"
    spec = importlib.util.spec_from_file_location("heartlib.heartcodec.models.encoder", enc_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.HeartCodecEncoder


def load(requested: str):
    from safetensors.torch import load_file

    HeartCodecEncoder = _import_encoder_cls()

    with open(CKPT / "encoder_config.json") as fh:
        cfg = json.load(fh)
    model = HeartCodecEncoder(cfg)
    state = load_file(str(CKPT / "encoder.safetensors"), device="cpu")
    state = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    device = requested
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    model.eval()
    return model, device, "HeartCodecEncoder standalone forward(audio_24k), continuous features"


TAIL = 240
STRIDE = 1920
MAX_SAMPLES = 30 * SR


def _valid_len(n: int) -> int:
    """Largest length <= min(n, 30 s) of the form k*1920 + 240."""
    cap = min(n, MAX_SAMPLES)
    k = (cap - TAIL) // STRIDE
    return max(k, 1) * STRIDE + TAIL


def _load_24k_stereo(wav_path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.T)  # (channels, samples)
    if sr != SR:
        from torchaudio.functional import resample
        x = resample(x, sr, SR)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)  # card: mono duplicated to stereo
    target = _valid_len(x.shape[-1])
    if x.shape[-1] >= target:
        x = x[:, :target]
    else:
        x = torch.nn.functional.pad(x, (0, target - x.shape[-1]))
    return x  # (2, target)


def extract(model, device: str, wav_path: Path):
    x = _load_24k_stereo(wav_path).to(device)        # (2, samples)
    with torch.no_grad():
        feats = model(x.unsqueeze(0))                # expects [batch, 2, samples]
    t = feats[0] if isinstance(feats, (tuple, list)) else feats
    t = t.detach()
    while t.dim() > 2:
        t = t.squeeze(0)
    return {"early": t, "middle": t, "final": t}


if __name__ == "__main__":
    run_smoke("heartcodec", load, extract, pick_clips(5), native_rate_hz=12.5)
