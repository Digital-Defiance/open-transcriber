"""Whisper base.en (MIT) load-and-extract smoke driver. Downloads the MIT checkpoint on first run."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import whisper

sys.path.insert(0, str(Path(__file__).parent))
from encoder_smoke import pick_clips, run_smoke  # noqa: E402

# base.en: 6 encoder blocks. Declared early/middle/final block outputs.
EARLY, MIDDLE = 1, 3  # 0-indexed block outputs; FINAL = post-LN encoder output


def load(requested: str):
    model = whisper.load_model("base.en", download_root="/Volumes/Code/copyist/.cache/whisper")
    device = requested
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    model.eval()
    return model, device, "whisper base.en, encoder blocks captured by forward hooks"


def extract(model, device: str, wav_path: Path):
    audio = whisper.load_audio(str(wav_path))          # 16 kHz mono float32
    audio = whisper.pad_or_trim(audio)                 # 30 s window -> 3000 mel frames
    mel = whisper.log_mel_spectrogram(audio).to(device)
    enc = model.encoder
    caps: dict[int, torch.Tensor] = {}
    handles = []
    for idx, block in enumerate(enc.blocks):
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, idx=idx: caps.__setitem__(idx, out.detach())))
    with torch.no_grad():
        final = enc(mel.unsqueeze(0))                  # (1, 1500, d_model); native 50 Hz over 30 s
    for h in handles:
        h.remove()
    return {
        "early": caps[EARLY].squeeze(0),
        "middle": caps[MIDDLE].squeeze(0),
        "final": final.squeeze(0),
    }


if __name__ == "__main__":
    run_smoke("whisper-base-en", load, extract, pick_clips(5), native_rate_hz=50.0)
