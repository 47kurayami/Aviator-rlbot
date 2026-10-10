# Training pipeline (scripted-teacher imitation)

Kept outside `src/` on purpose: training-only code must never enter the validated
runtime sandbox (`src/` forbids `socket`/`subprocess`/`requests`/`urllib` and
`eval`/`exec`). Nothing here runs during a match submission.

Pipeline:

```
record_teacher.py  ->  training/datasets/raw/*.jsonl
build_dataset.py   ->  training/datasets/processed/*.npz + splits.json + dataset_manifest.json
train.py           ->  training/runs/<timestamp>/{best.pt,last.pt,metrics.json,provenance.json}
```

`training/datasets/` and `training/runs/` are ignored by git and skipped by the
kit's file manifest (the kit excludes any dir named `datasets` or `runs`).

## 1. Record teacher data (Windows + RLBot)

The recorder needs the repo's `.venv` (run `setup.ps1` first). In the RLBot GUI:

1. Add File -> `training/bot.toml` (a separate bot from the repo-root `bot.toml`).
2. Put **TeacherRecorder** on one side, an opponent on the other, Soccar 1v1.
3. Start Match. Play ~2-3 matches first, then copy the `.jsonl` files out of
   `training/datasets/raw/` as a backup before scaling up.

One process == one match == one JSONL file. Each line is one processed callback:

```json
{"v":1,"match_id":"...","callback":1,"frame":2,"elapsed":0.0166,"index":0,"team":0,
 "obs":[13 floats],"controls":[throttle,steer,pitch,yaw,roll,jump,boost,handbrake]}
```

`obs` is produced through the same `live_adapter(...)[:13]` path the shipped
runtime uses, so the student sees identical features. `training/bot.toml`
sets `--teacher smart`, which adds purposeful jump/flip events — grounded,
aimed at a close ball, on the side that sends it toward the opponent goal — so
the teacher keeps playing well while producing flip examples; the default
`reference` teacher matches `reference/python_example/src/bot.py`.

## 2. Build the dataset

```bash
python training/build_dataset.py --raw training/datasets/raw --out training/datasets/processed \
    --version scripted-teacher-v1 --teacher reference
```

Splits are assigned by **whole match**, before any fitting. The manifest records
class histograms and SHA-256 of every raw/processed file.

## 3. Train

```bash
python training/train.py --data training/datasets/processed --out training/runs \
    --epochs 30 --chunk 120 --steer-weight 1.0
```

- Frozen architecture: `Policy` from `src/policy.py`; asserts 26,181 parameters.
- One recurrent stream per match; hidden detached between TBPTT chunks; reset
  only at match boundaries.
- Loss: mode cross-entropy (class-weighted by default) + Chase-masked steering MSE.
- Device auto-detects CUDA → Apple MPS → CPU, so the same command is fast on a
  Mac GPU and on Windows.
- `best.pt` is selected by lowest validation loss. It is a plain
  `{"state_dict": ...}`, ready for `kit.py seal`.

Watch per-class recall for `Jump` / `Front dodge` and the flip hit-rate/latency,
not just aggregate accuracy.

## 3b. Calibrate the rare-class bias (fast, no retrain)

Class-weighted CE makes the policy over-predict the up-weighted rare classes
(the sealed run hopped constantly). `recalibrate.py` removes that bias by
subtracting `alpha*log(class_weight)` from the four mode logits, folded into the
existing `head.bias` — no shape or parameter change:

```bash
python training/recalibrate.py --checkpoint training/runs/<ts>/best.pt \
    --data training/datasets/processed \
    --out training/runs/<ts>/best_calibrated.pt --alpha 0.35
```

`alpha=1.0` fully removes the weighting bias (can zero out Jump); `alpha≈0.35`
puts predicted Jump/Front-dodge back on the teacher's true base rate. It writes
`provenance_calibrated.json`, sealable directly. Runs on macOS/Windows/Linux.

## 4. Seal, preflight, submit

```bash
python tools/kit.py seal --checkpoint training/runs/<timestamp>/best.pt \
    --provenance training/runs/<timestamp>/provenance.json
python src/preflight.py
```

`train.py` writes `provenance.json` with the actual dataset version, seed,
splits, objective, per-class metrics, flip metrics and environment versions; it
satisfies the seal requirement directly. Then follow `docs/TESTING.md` for the
live GUI ladder and update `docs/PARTICIPANT_RESULTS.md`.

## Offline smoke test (no game)

```bash
python training/make_synthetic.py --matches 6 --frames 400
python training/build_dataset.py --raw training/datasets/raw --out training/datasets/processed
python training/train.py --epochs 2 --chunk 40 --out training/runs/smoke
```

## Honest limits

- Offline metrics are not closed-loop gameplay. A learner changes the states it
  later sees, so teacher-forced validation overstates competence. Run a live GUI
  match before promoting any model.
- The plain scripted teacher only exercises the four shipped modes and rarely
  flips. `--teacher smart` improves flip coverage by flipping only when aimed at
  a close ball; it deviates from the raw reference behaviour and must be
  disclosed as an augmentation. Verify it still beats the same opponents before
  trusting the data (an indiscriminate flip teacher plays *worse*).
- The reference scripted bot must never be used as a live fallback controller.
