"""The identical training loop for every encoder probe: fixed optimizer/schedule/early-stop, three seeds.

Deterministic given a seed. Early stopping uses **only** allowed synthetic / CC BY validation (sampler val +
VocalSet val singers), never real dev. Per-epoch checkpoints are written atomically so a long run resumes.
The multi-task loss sums the per-head losses with fixed weights.

A ``Batch`` is a small dict of tensors so the loop is CPU-testable with tiny synthetic data; the GPU window
fills batches from extracted features and the sampler/SoulX labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .model import ProbeConfig, ProbeModel

SEEDS = (20261020, 20261021, 20261022)
#: Fixed multi-task loss weights.
LOSS_WEIGHTS = {"pitch": 1.0, "silence": 0.5, "onset": 1.0, "offset": 0.5,
                "beat": 0.5, "downbeat": 0.5, "chord_root": 0.5, "chord_quality": 0.5, "key": 0.5}


@dataclass
class TrainConfig:
    probe: ProbeConfig
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5          # early-stop after this many epochs with no val improvement
    seed: int = SEEDS[0]


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except Exception:
        pass


def multitask_loss(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum the per-head losses (BCE for frame pulses/silence, CE for pitch/chord/key)."""
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    total = out["pitch"].new_zeros(())
    for head, w in LOSS_WEIGHTS.items():
        if head not in targets:
            continue
        pred, tgt = out[head], targets[head]
        if head in ("pitch", "chord_root", "chord_quality"):
            total = total + w * ce(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1).long())
        elif head == "key":
            total = total + w * ce(pred, tgt.long())
        else:  # silence, onset, offset, beat, downbeat: per-frame binary
            total = total + w * bce(pred, tgt.float())
    return total


def _epoch(net: ProbeModel, batches: list[dict], optim, train: bool) -> float:
    net.train(train)
    total, n = 0.0, 0
    for b in batches:
        if train:
            optim.zero_grad()
        out = net(b["layers"], b["bp"])
        loss = multitask_loss(out, b["targets"])
        if train:
            loss.backward()
            optim.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


@dataclass
class TrainResult:
    seed: int
    best_val: float
    epochs_run: int
    history: list[float] = field(default_factory=list)


def train_probe(cfg: TrainConfig, train_batches: list[dict], val_batches: list[dict],
                checkpoint_dir: Path | None = None) -> tuple[ProbeModel, TrainResult]:
    """Train deterministically; early-stop on val loss; checkpoint the best model. Returns (model, result)."""
    seed_everything(cfg.seed)
    net = ProbeModel(cfg.probe)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(cfg.max_epochs):
        _epoch(net, train_batches, optim, train=True)
        with torch.no_grad():
            val = _epoch(net, val_batches, optim, train=False)
        history.append(val)
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if checkpoint_dir is not None:
                _checkpoint(checkpoint_dir, cfg.seed, epoch, best_state, val)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, TrainResult(seed=cfg.seed, best_val=best_val, epochs_run=len(history), history=history)


def _checkpoint(directory: Path, seed: int, epoch: int, state: dict, val: float) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"probe-seed{seed}-best.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"state_dict": state, "epoch": epoch, "val": val, "seed": seed}, tmp)
    tmp.replace(path)
    meta = directory / f"probe-seed{seed}-best.json"
    meta.write_text(json.dumps({"epoch": epoch, "val": val, "seed": seed}, indent=1) + "\n", encoding="utf-8")
    return path
