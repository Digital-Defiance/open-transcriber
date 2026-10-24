#!/usr/bin/env python3
"""Establish the CEILING and reach a VERDICT (CPU, no GPU, no re-singing).

Uses the pilot's 200 cached separated vocals + 42 cached dev stems. Steps:
  1. Build per-song features (unique cache key) and labels aligned three ways: raw (score time), linear
     rescale, DTW. Report BP-passthrough pc_on50 on TRAIN for each (the representational ceiling).
  2. Train a FrameModel on the DTW-aligned data.
  3. Score the trained model on REAL dev (rwc-other-dev, karaoke-dev) through the fair cross-fit eval path,
     and also report BP-passthrough on real dev (the floor the learned model should beat on its own inputs).
Decision bar: a trained-or-passthrough number approaching the BP baseline (~22.9 RWC / ~36.7 karaoke)
justifies a GPU retrain; stuck well below (<~15) is the honest negative -> stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import paths, score  # noqa: E402
from open_transcriber.board import board  # noqa: E402
from open_transcriber.data import frames  # noqa: E402
import scripts.run_frames as rf  # noqa: E402

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 160
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def song_vocal(slug, out):
    return out / "frames-run" / "sep" / slug / "htdemucs" / "mix" / "vocals.wav"


def passthrough_f1(feats, truth):
    yn, yo = frames.rasterise_labels(truth, len(feats))
    ref = frames.decode_notes(yn, yo, 0.5, 0.5)
    est = frames.decode_notes(feats[:, :frames.N_BINS], feats[:, frames.N_BINS:2 * frames.N_BINS])
    return score.score_pair(ref, est, ("pc_on50",)).get("pc_on50_f1", 0.0)


def main() -> int:
    out = paths.out_dir(); root = paths.copyist_root(); data = paths.data_dir()
    run = out / "frames-run"
    bp_dir = run / "train-bp-fixed"
    slugs = [d.name for d in sorted((run / "sep").iterdir()) if d.is_dir()][:N_TRAIN]

    print(f"=== loading {len(slugs)} cached sung clips, 3 label alignments ===")
    data_raw, data_lin, data_dtw = [], [], []
    pt_raw, pt_lin, pt_dtw = [], [], []
    for slug in slugs:
        wav = song_vocal(slug, out)
        if not wav.exists():
            continue
        feats = rf.features_for_wav(root, wav, bp_dir, key=slug)
        truth = rf._midi_truth(root / "out" / slug / "stems" / "midi" / "melody.mid")
        bpn = feats[:, :frames.N_BINS]
        t_lin = frames.align_truth_to_audio(truth, bpn)
        t_dtw = frames.align_truth_dtw(truth, bpn)
        for store, pt, tr in ((data_raw, pt_raw, truth), (data_lin, pt_lin, t_lin), (data_dtw, pt_dtw, t_dtw)):
            yn, yo = frames.rasterise_labels(tr, len(feats))
            store.append((feats, yn, yo))
            pt.append(passthrough_f1(feats, tr))
    print(f"  BP-passthrough TRAIN pc_on50  raw={100*np.mean(pt_raw):.1f}  "
          f"linear={100*np.mean(pt_lin):.1f}  DTW={100*np.mean(pt_dtw):.1f}  (the alignment ceiling)")

    print(f"\n=== training FrameModel on DTW-aligned data ({len(data_dtw)} clips, {EPOCHS} epochs) ===")
    model = frames.FrameModel(hidden=64, seed=0)
    info = model.fit(data_dtw, epochs=EPOCHS)
    tr_f1 = []
    for f, yn, yo in data_dtw:
        ref = frames.decode_notes(yn, yo, 0.5, 0.5)
        pn, po = model.predict(f)
        tr_f1.append(score.score_pair(ref, frames.decode_notes(pn, po), ("pc_on50",)).get("pc_on50_f1", 0.0))
    print(f"  loss={info['loss']:.3f}  TRAIN pc_on50 (model vs own DTW labels)={100*np.mean(tr_f1):.1f}")

    print("\n=== REAL-DEV through the fair cross-fit eval path (the decision number) ===")
    model.save(run / "model-dtw.pt")
    feats_by_split = rf.dev_features(root, out, data)
    dec = {"frame_thresh": 0.3, "onset_thresh": 0.5}
    ev = rf.eval_real_dev(model, feats_by_split, data, out, dec, calibrate=True)
    for split, r in ev.items():
        print(f"  {split}: MODEL pc_on50={r['model_pc_on50']}  BASELINE(BP one line)={r['baseline_pc_on50']}  "
              f"shift model/base={r['model_shift']}/{r['baseline_shift']}  n={r['n']}")

    # BP-passthrough on real dev (what the learned model should at least match, since BP is its input)
    print("\n=== BP-passthrough on REAL dev (reference floor) ===")
    for split in ("rwc-other-dev", "karaoke-dev"):
        kind = board.SET_METRICS[split][0]
        rows = []
        for item in board.set_members(split):
            feats = feats_by_split.get(split, {}).get(item)
            if feats is None:
                continue
            est = frames.decode_notes(feats[:, :frames.N_BINS], feats[:, frames.N_BINS:2 * frames.N_BINS], **dec)
            rows.append(board._score_item(split, kind, item, est, data))
        print(f"  {split}: BP-passthrough pc_on50={100*score.mean_f1(rows,'pc_on50'):.1f}  n={len(rows)}")

    (out / "ceiling.json").write_text(json.dumps(
        {"passthrough_train": {"raw": 100*np.mean(pt_raw), "linear": 100*np.mean(pt_lin),
                               "dtw": 100*np.mean(pt_dtw)},
         "model_train_dtw": 100*np.mean(tr_f1), "real_dev": ev, "n_train": len(data_dtw),
         "epochs": EPOCHS}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
