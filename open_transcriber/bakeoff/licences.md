# Encoder licence table (bake-off, committed in the package)

This file is the worktree-safe licence gate for `run_bakeoff.licences_confirmed`. A worktree has no tracked
`.agents/`, so the gate reads this package file by default. Every tested encoder is listed with its pinned
source, code licence, weight licence and a free/allowed verdict. Only free, permissive encoders are used;
any new model weights Copyist trains here are **MIT**. RWC and the karaoke songs stay CC BY-NC
(evaluation/probe-eval only, never training). Re-check each model card at pull time in the GPU window.
Content was rephrased for compliance with licensing restrictions.

| Encoder | Pinned source | Code licence | Weight licence | Free/allowed | Notes |
|---|---|---|---|---|---|
| **MusicFM** (FMA checkpoint) | huggingface.co/minzwon/MusicFM; github.com/minzwon/musicfm | MIT (one Apache-2.0 flash-conformer file) | FMA checkpoint, trained on Creative Commons audio | **Yes** | Reference extractor. 24 kHz mono, hidden layers 3/7/12, (frames,1024) @ 25 Hz on mps. |
| **Dasheng base** | huggingface.co/mispeech/dasheng-base; github.com/XiaoMi/dasheng; PyPI `dasheng` | Apache-2.0 | Apache-2.0 | **Yes** | base only. Final (frames,768) @ 25 Hz on mps; early/middle only via the real hidden-states API, else final-only. |
| **HeartCodec encoder** | huggingface.co/HeartMuLa/HeartCodec-oss-encoder; github.com/HeartMuLa/heartlib | Apache-2.0 | Apache-2.0 | **Yes (but DROPPED)** | Licence-clean, but **dropped: unsupported on MPS without a clear pilot lead** (24k->16k resample exceeds the MPS 65536-channel conv1d limit). Recorded in the drop ledger; not instantiated for extraction. |
| **Whisper `base.en`** | github.com/openai/whisper | MIT | MIT (code and weights) | **Yes** | Auxiliary ear per the prereg Whisper rule. Encoder blocks 1/3/6, (1500,512) @ 50 Hz, CPU only (mps blocked by a SparseMPS op). |
| **Basic Pitch** posteriors | github.com/spotify/basic-pitch | Apache-2.0 | Apache-2.0 | **Yes** | The fixed melody side input and the encoder-free control. Runs from `.venv` (TensorFlow) across a subprocess boundary; never imported into the ear env. |

## Verdicts

- All five are free and MIT/Apache-2.0 (plus the FMA checkpoint's CC-licensed training data). None carries a
  non-commercial or no-derivatives restriction, and none is ambiguous, so none is blocked on licence grounds.
- HeartCodec's only flag is **operational, not legal**: its published path targets CUDA and the MPS resample
  hits a hard conv1d limit, so the prereg compute rule drops it. The drop is recorded in the run ledger.
- New probe/head weights trained here are **MIT**.

## Recheck-at-download checklist (GPU window)

1. Re-open each model card and confirm the licence field is unchanged from the table above.
2. Confirm the exact revision/commit pinned for each checkpoint and record its hash in `runs.jsonl`.
3. Fail closed: if any licence reads as non-commercial, no-derivatives, or ambiguous at pull time, skip that
   encoder and note it.
4. Stay above the 150 GB free-disk floor.
