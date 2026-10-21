"""Dasheng-base (Apache-2.0) load-and-extract smoke driver. Weights from Zenodo via the dasheng package."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from encoder_smoke import pick_clips, run_smoke  # noqa: E402

import dasheng  # noqa: E402

EARLY, MIDDLE = 2, 6  # 0-indexed block outputs of 12; FINAL = model output


def load(requested: str):
    model = dasheng.dasheng_base()
    device = requested
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    model.eval()
    return model, device, "dasheng_base, 12 transformer blocks, raw 16k waveform input"


def _load_16k_mono(wav_path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != 16000:
        import torchaudio
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16000).numpy()
    return torch.from_numpy(mono).unsqueeze(0)  # (1, samples)


def extract(model, device: str, wav_path: Path):
    x = _load_16k_mono(wav_path).to(device)
    caps: dict[int, torch.Tensor] = {}
    handles = []
    for idx, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, idx=idx: caps.__setitem__(idx, (out[0] if isinstance(out, tuple) else out).detach())))
    with torch.no_grad():
        final = model(x)  # (1, frames, 768)
    for h in handles:
        h.remove()
    return {
        "early": caps[EARLY].squeeze(0),
        "middle": caps[MIDDLE].squeeze(0),
        "final": final.squeeze(0),
    }


if __name__ == "__main__":
    run_smoke("dasheng-base", load, extract, pick_clips(5), native_rate_hz=25.0)
