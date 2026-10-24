#!/usr/bin/env python3
"""CPU diagnosis of the sung frame-model's implausible ~1% real-dev result. No GPU, no retraining.

Probes, with numbers:
  A. Does the TRAINED model reproduce notes on its own TRAIN clips (known labels)?  -> decode vs domain bug.
  B. Predicted note counts per real-dev clip vs reference counts + activation stats. -> near-empty / threshold.
  C. BP-posterior passthrough control (same decode on the raw BP features).          -> is decode itself sane?
  D. Does the model light up on dev at all (activation distributions train vs dev)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import paths, score  # noqa: E402
from open_transcriber.board import board  # noqa: E402
from open_transcriber.data import frames  # noqa: E402
import scripts.run_frames as rf  # noqa: E402


def ref_notes(split, item, data):
    if board.SET_METRICS[split][0] == "rwc":
        ref, _sung, _exc = board.load_rwc_reference(item, data)
        return ref
    return board.load_karaoke_reference(item, data)


def main() -> int:
    out = paths.out_dir()
    data = paths.data_dir()
    root = paths.copyist_root()
    run_dir = out / "frames-run"
    model = frames.FrameModel.load(run_dir / "model.pt")
    print(f"model loaded: hidden={model.hidden} seed={model.seed}")

    # --- A. reproduce on TRAIN clips ------------------------------------------------------------------
    clips = sorted((run_dir / "clips").glob("*.npz"))[:12]
    print(f"\n=== A. TRAIN-clip reproduction ({len(clips)} clips; score est vs the clip's OWN labels) ===")
    f1s = []
    for cp in clips:
        blob = np.load(cp)
        feats, yn, yo = blob["feats"], blob["yn"], blob["yo"]
        ref = frames.decode_notes(yn, yo, frame_thresh=0.5, onset_thresh=0.5)   # invert labels -> notes
        pn, po = model.predict(feats)
        est = frames.decode_notes(pn, po)
        row = score.score_pair(ref, est, ("pc_on50",))
        f1 = row.get("pc_on50_f1", 0.0)
        f1s.append(f1)
        print(f"  {cp.stem}: label_notes={len(ref.onset):3d} pred_notes={len(est.onset):3d} "
              f"note_act_max={pn.max():.3f} mean={pn.mean():.4f} onset_act_max={po.max():.3f} "
              f"pc_on50={100*f1:.1f}")
    print(f"  TRAIN mean pc_on50 (est vs own labels): {100*np.mean(f1s):.1f}")

    # --- B. real-DEV predicted vs reference counts + the canonical scorer -----------------------------
    print("\n=== B. real-DEV predicted vs reference note counts, and the canonical _score_item ===")
    for split in ("rwc-other-dev", "karaoke-dev"):
        kind = board.SET_METRICS[split][0]
        for item in board.set_members(split)[:6]:
            wav = rf.dev_vocal_wav(item, out, data)
            if wav is None:
                print(f"  {item}: NO WAV"); continue
            feats = rf.features_for_wav(root, wav, run_dir / "dev-bp")
            pn, po = model.predict(feats)
            est = frames.decode_notes(pn, po)
            ref = ref_notes(split, item, data)
            row = board._score_item(split, kind, item, est, data)
            print(f"  {item}: frames={len(feats):5d} pred_notes={len(est.onset):3d} ref_notes={len(ref.onset):3d} "
                  f"note_act_max={pn.max():.3f} mean={pn.mean():.4f} onset_max={po.max():.3f} "
                  f"pc_on50={100*row.get('pc_on50_f1',0.0):.1f}")

    # --- C. BP-passthrough control (same decode on raw BP features) -----------------------------------
    print("\n=== C. control: decode the RAW BP posteriors directly (same decode fn) ===")
    for item in board.set_members("rwc-other-dev")[:4]:
        wav = rf.dev_vocal_wav(item, out, data)
        if wav is None:
            continue
        feats = rf.features_for_wav(root, wav, run_dir / "dev-bp")
        bp_note = feats[:, :frames.N_BINS]
        bp_onset = feats[:, frames.N_BINS:2 * frames.N_BINS]
        est_bp = frames.decode_notes(bp_note, bp_onset)
        print(f"  {item}: BP-passthrough notes={len(est_bp.onset)} "
              f"bp_note_max={bp_note.max():.3f} bp_onset_max={bp_onset.max():.3f}")

    # --- D. activation distributions: train vs dev ----------------------------------------------------
    print("\n=== D. note-head activation: train vs dev (does the model respond on dev?) ===")
    tr = np.load(clips[0])["feats"]
    ptn, _ = model.predict(tr)
    print(f"  train clip: note_act max={ptn.max():.3f} p99={np.percentile(ptn,99):.3f} mean={ptn.mean():.4f}")
    w0 = rf.dev_vocal_wav(board.set_members('rwc-other-dev')[0], out, data)
    pdn, _ = model.predict(rf.features_for_wav(root, w0, run_dir / "dev-bp"))
    print(f"  dev clip:   note_act max={pdn.max():.3f} p99={np.percentile(pdn,99):.3f} mean={pdn.mean():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
