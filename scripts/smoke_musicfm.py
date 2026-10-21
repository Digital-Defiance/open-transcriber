"""MusicFM FMA checkpoint (MIT) load-and-extract smoke driver.

Repo cloned to .cache/ear/encoders/musicfm; FMA checkpoint (MIT, CC-trained) in its data/ dir. The package
uses absolute ``musicfm.*`` imports, so the clone's PARENT dir goes on sys.path and we import the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from encoder_smoke import pick_clips, run_smoke  # noqa: E402

REPO = Path("/Volumes/Code/copyist/.cache/ear/encoders")
sys.path.insert(0, str(REPO))  # so `import musicfm.model...` resolves the clone
from musicfm.model.musicfm_25hz import MusicFM25Hz  # noqa: E402

DATA = REPO / "musicfm" / "data"
SR = 24000
EARLY, MIDDLE, FINAL = 3, 7, 12  # 25 Hz hidden states 0..12


def load(requested: str):
    model = MusicFM25Hz(stat_path=str(DATA / "fma_stats.json"), model_path=str(DATA / "pretrained_fma.pt"))
    device = requested
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    model.eval()
    return model, device, "MusicFM25Hz FMA ckpt, get_latent hidden states, 24k input"


def _load_24k_mono(wav_path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SR:
        import torchaudio
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR).numpy()
    return torch.from_numpy(mono).unsqueeze(0)


def extract(model, device: str, wav_path: Path):
    x = _load_24k_mono(wav_path).to(device)
    out = {}
    with torch.no_grad():
        for name, ix in (("early", EARLY), ("middle", MIDDLE), ("final", FINAL)):
            emb = model.get_latent(x, layer_ix=ix)   # (1, frames, dim)
            out[name] = emb.squeeze(0).detach()
    return out


if __name__ == "__main__":
    run_smoke("musicfm", load, extract, pick_clips(5), native_rate_hz=25.0)
