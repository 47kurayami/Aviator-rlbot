# Participant results — Kurayami (`aviator/kurayami`)

## Checkpoint identity

| Item | Value |
|---|---|
| Shipped weights | `model/V13.pt` |
| SHA-256 | `f387c1c219726bae658d5451ccf79248c13a9ecc46bc94f815c7a38e94ed4169` |
| Training run | `training/runs/20261011_000530/best.pt` (best epoch 44 / 50) |
| Starter weights (reference) | `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7` |
| Parameters | 26,181 (13D → Linear(13,64)+ReLU → GRU(64,64) → Linear(64,5)) |
| Cosine similarity vs starter | 0.4626 (well below the 0.999 anti-noise gate) |

Architecture, the 13-D observation contract, and the action decoder are
**unchanged**. Only the weight values were replaced by trained weights.

## What changed and how it was trained

Method: **scripted-teacher behaviour cloning** (imitation learning).

1. `training/record_teacher.py` ran the RLBot python-example scripted teacher
   (`--teacher smart`) and logged, per processed callback, the exact 13-D student
   observation (the same `live_adapter(...)[:13]` path `src/runtime.py` uses) plus
   the teacher's raw controls.
2. `training/build_dataset.py` turned controls into `(mode, steer)` labels with
   whole-match train/val/test splits (seed 42).
3. `training/train.py` fit the frozen `Policy` with **mildly** class-weighted mode
   cross-entropy (inverse-frequency weights raised to power 0.5) plus Chase-masked
   steering MSE (Adam, lr 3e-4, TBPTT chunk 120, gradient clipping 1.0, hidden
   reset only at match boundaries). `best.pt` is chosen by lowest validation loss.
4. `tools/kit.py seal` re-verified the frozen shapes, copied the checkpoint to
   `model/V13.pt`, and recorded the full record in `submission.json` →
   `training_provenance`.

Data: 34 recorded 1v1 matches (`teacher-v3`), 593,822 callback frames; split
**24 train / 5 val / 5 test** by whole match. Raw/processed data and virtual
environments are **not** shipped.

Environment: Python 3.12.10, torch 2.6.0+cu124 (NVIDIA RTX 4060), numpy 2.5.3.

Commands:

```
python training/build_dataset.py --raw training/datasets/raw \
    --out training/datasets/processed_v3 --version teacher-v3 --teacher smart
python training/train.py --data training/datasets/processed_v3 \
    --out training/runs --epochs 50 --chunk 120 \
    --class-weight-power 0.5 --grad-clip 1.0
python tools/kit.py seal \
    --checkpoint training/runs/20261011_000530/best.pt \
    --provenance training/runs/20261011_000530/provenance.json
```

## Class weighting (why power 0.5)

The earlier `teacher-v2` run used full inverse-frequency class weights
(Jump 30.7×, Front dodge 9.5×). That over-predicted Jump ~4.8× and made the bot
hop constantly in the live GUI, which forced a post-hoc calibration
(`recalibrate.py`, α=0.35) to undo the bias in `head.bias`.

For `teacher-v3` the weights are **tempered at the source** instead: each weight
is `(N / (4·counts))^0.5` (Jump ≈ 5.1×, Front dodge ≈ 2.8×), and gradient
clipping stabilised the tail of training. No recalibration step is needed —
the predicted rare-class rates land on the teacher's true base rates.

## Measured offline results (teacher-forced)

All rows below are evaluated on the **same** `teacher-v3` validation split
(97,522 frames; true counts: Neutral 28,376, Chase 65,960, Jump 748, Front
dodge 2,438) so they are directly comparable. Predicted-action counts:

| Version | Neutral | Chase | Jump | Front dodge | steer MAE (Chase) | flip hit |
|---|---|---|---|---|---|---|
| `teacher-v2` base (full weights) | 21,755 | 70,323 | 2,343 | 3,101 | — | — |
| `teacher-v2` calibrated (previous ship) | 22,043 | 72,928 | 388 | 2,163 | — | 0.536¹ |
| **`teacher-v3` (this checkpoint)** | 27,170 | 67,431 | **752** | 2,169 | 0.206 | 0.619 |

¹ Front-dodge recall of the calibrated `teacher-v2`; its Jump recall was 0.067.

Test split (97,081 frames; true Jump 643, Front dodge 2,119): predicted Jump
**639**, Front dodge 1,806; steering MAE 0.172; flip hit-rate **0.710**
(Jump 0.578, Front dodge 0.849).

Interpretation: the `teacher-v3` model predicts Jump at almost exactly the
teacher's rate (752 vs 748 on val; 639 vs 643 on test) with much higher Jump
precision than the `teacher-v2` base, and stronger Front-dodge recall (≈0.77 vs
≈0.54 for the calibrated `teacher-v2`). Aggregate accuracy was never the target —
per-class behaviour is. Full confusion matrices are in `submission.json` →
`training_provenance`.

## Live GUI testing

- **`teacher-v2` base (pre-calibration):** spawned and drove under the RLBot GUI
  in real 1v1 matches; it chased but **hopped excessively** and did **not**
  front-flip. This motivated the weighting fix above.
- **`teacher-v2` calibrated (previous ship):** hop spam removed offline; live
  ladder not completed.
- **`teacher-v3` (this ship):** **NOT YET TESTED live.** Run the `docs/TESTING.md`
  ladder (vs untouched V13 / Beginner / Rookie) and update this section. Offline
  metrics are not closed-loop proof.

## Known limits

- Offline imitation metrics are not closed-loop gameplay; a learned policy shifts
  the states it sees (covariate shift), so live play will differ from validation.
- Front flips require a rare 3-frame `Jump → release → Jump+forward` sequence;
  behaviour cloning rarely reproduces it, so flips likely stay sparse.
- The scripted `smart` teacher is an augmentation, not the raw reference; flip
  labels are sparse. No defence or opponent modelling beyond the imitation data.
- Training code lives only in `training/` (outside the scanned `src/` sandbox).
