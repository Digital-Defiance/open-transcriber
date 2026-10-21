"""The identical probe body and multi-task heads for every encoder (pure PyTorch, CPU-testable).

Per the pre-registration: a learned 3-way scalar mix over the encoder's declared early/middle/final layers;
resample to 50 Hz; a linear projection to 256; a 2-layer BiGRU (width 192, dropout 0.1); 12 s windows with
fixed overlap. Heads: vocal pitch/silence, note onset, note offset, beat, downbeat, chord root + maj/min/
no-chord, 24-key. Basic Pitch contour/note/onset posteriors are concatenated into every melody head's input,
and the control configuration uses Basic Pitch posteriors only (no encoder).

Every shape is a constructor argument so the whole model runs on CPU with tiny synthetic batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

TARGET_RATE_HZ = 50.0
WINDOW_SECONDS = 12.0
#: Head output widths.
N_PITCH = 128          # vocal pitch classes (MIDI 0..127); silence handled by a separate logit
N_CHORD_ROOT = 13      # 12 roots + no-chord
N_CHORD_QUALITY = 3    # maj / min / no-chord
N_KEY = 24             # 12 major + 12 minor
#: Basic Pitch posterior width concatenated into the melody heads (contour/note/onset summarised to 3 chans
#: here; the real extractor supplies the full posterior stack, this is the tested interface).
BP_POSTERIOR_DIM = 3


@dataclass(frozen=True)
class ProbeConfig:
    feature_dim: int               # per-layer encoder feature width
    n_layers: int = 3              # early/middle/final
    proj_dim: int = 256
    gru_hidden: int = 192
    gru_layers: int = 2
    dropout: float = 0.1
    bp_dim: int = BP_POSTERIOR_DIM
    control: bool = False          # control = Basic Pitch posteriors only, no encoder features


def resample_time(x: torch.Tensor, native_rate: float, target_rate: float = TARGET_RATE_HZ) -> torch.Tensor:
    """Resample a (batch, frames, dim) sequence along time to ``target_rate`` with linear interpolation."""
    if abs(native_rate - target_rate) < 1e-6:
        return x
    b, t, d = x.shape
    new_t = max(1, int(round(t * target_rate / native_rate)))
    # interpolate expects (batch, channels, length)
    y = F.interpolate(x.transpose(1, 2), size=new_t, mode="linear", align_corners=False)
    return y.transpose(1, 2)


class LayerMix(nn.Module):
    """A learned 3-way scalar mix (softmax weights) over the declared encoder layers."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        # layers: (batch, n_layers, frames, dim)
        w = torch.softmax(self.weights, dim=0).view(1, -1, 1, 1)
        return (layers * w).sum(dim=1)


class ProbeModel(nn.Module):
    """Shared body + heads. Input: stacked encoder layers at 50 Hz and Basic Pitch posteriors."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.mix = LayerMix(cfg.n_layers)
        body_in = (0 if cfg.control else cfg.feature_dim) + cfg.bp_dim
        self.proj = nn.Linear(body_in, cfg.proj_dim)
        self.gru = nn.GRU(cfg.proj_dim, cfg.gru_hidden, num_layers=cfg.gru_layers,
                          batch_first=True, bidirectional=True, dropout=cfg.dropout)
        h = cfg.gru_hidden * 2
        # melody heads also see the Basic Pitch posteriors directly (concatenated)
        mel_in = h + cfg.bp_dim
        self.pitch_head = nn.Linear(mel_in, N_PITCH)
        self.silence_head = nn.Linear(mel_in, 1)
        self.onset_head = nn.Linear(mel_in, 1)
        self.offset_head = nn.Linear(mel_in, 1)
        # rhythm / harmony / key heads see the body output
        self.beat_head = nn.Linear(h, 1)
        self.downbeat_head = nn.Linear(h, 1)
        self.chord_root_head = nn.Linear(h, N_CHORD_ROOT)
        self.chord_quality_head = nn.Linear(h, N_CHORD_QUALITY)
        self.key_head = nn.Linear(h, N_KEY)

    def forward(self, layers: torch.Tensor, bp: torch.Tensor) -> dict[str, torch.Tensor]:
        """layers: (batch, n_layers, frames, feature_dim); bp: (batch, frames, bp_dim) at 50 Hz.

        Returns per-frame logits for every head (key is pooled over time).
        """
        if self.cfg.control:
            body_in = bp
        else:
            mixed = self.mix(layers)                       # (batch, frames, feature_dim)
            body_in = torch.cat([mixed, bp], dim=-1)
        z = torch.relu(self.proj(body_in))
        seq, _ = self.gru(z)                               # (batch, frames, 2*hidden)
        mel_feats = torch.cat([seq, bp], dim=-1)
        out = {
            "pitch": self.pitch_head(mel_feats),
            "silence": self.silence_head(mel_feats).squeeze(-1),
            "onset": self.onset_head(mel_feats).squeeze(-1),
            "offset": self.offset_head(mel_feats).squeeze(-1),
            "beat": self.beat_head(seq).squeeze(-1),
            "downbeat": self.downbeat_head(seq).squeeze(-1),
            "chord_root": self.chord_root_head(seq),
            "chord_quality": self.chord_quality_head(seq),
            "key": self.key_head(seq.mean(dim=1)),         # one 24-key logit per clip
        }
        return out


def stack_layers(features: dict, order=("early", "middle", "final")) -> torch.Tensor:
    """Stack an extractor's per-layer feature arrays into (1, n_layers, frames, dim) as a float tensor."""
    arrs = [torch.as_tensor(features[name], dtype=torch.float32) for name in order]
    return torch.stack(arrs, dim=0).unsqueeze(0)
