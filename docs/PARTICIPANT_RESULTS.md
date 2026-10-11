# Participant results — Kurayami (`aviator/kurayami`)

## Checkpoint identity

| Item | Value |
|---|---|
| Shipped weights | `model/V13.pt` |
| SHA-256 | `4dcb952f47eb585d4d9d243c5ecefee611ab9d55d87a2e337578b6bcbe8117b8` |
| Training run | `training/runs/20261011_025357/best.pt` (best epoch 47 / 50) |
| Starter weights (reference) | `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7` |
| Parameters | 26,181 (13D → Linear(13,64)+ReLU → GRU(64,64) → Linear(64,5)) |
| Cosine similarity vs starter | 0.4196 (well below the 0.999 anti-noise gate) |

Architecture, the 13-D observation contract, and the action decoder are
**unchanged**. Only the weight values were replaced by trained weights.

## What changed and how it was trained

Method: **scripted-teacher imitation learning with DAgger** (dataset-aggregation
on-policy rounds).

1. `training/record_teacher.py` ran the RLBot python-example scripted teacher
   (`--teacher smart`) and logged, per processed callback, the exact 13-D student
   observation (the same `live_adapter(...)[:13]` path `src/runtime.py` uses) plus
   the teacher's raw controls.
2. `training/build_dataset.py` turned controls into `(mode, steer)` labels with
   whole-match train/val/test splits (seed 42).
3. `training/train.py` fit the frozen `Policy` with mildly class-weighted mode
   cross-entropy plus Chase-masked steering MSE (Adam, lr 3e-4, TBPTT chunk 120,
   gradient clipping 1.0, hidden reset only at match boundaries). `best.pt` is
   chosen by lowest validation loss.
4. **DAgger:** `training/record_dagger.py` let the trained *student* drive live
   1v1 matches against Pro while the scripted teacher labelled every
   student-visited state, logging both. Those on-policy frames were added to the
   dataset and the model was retrained with a milder rare-class weighting
   (power 0.35) to keep flip over-prediction in check.
5. `tools/kit.py seal` re-verified the frozen shapes, copied the checkpoint to
   `model/V13.pt`, and recorded the full record in `submission.json` →
   `training_provenance`.

Data: **51 matches** (scripted-teacher recordings plus DAgger on-policy matches);
train split 41 matches / 659,430 frames, val 5 matches / 97,522 frames, test
5 matches / 97,081 frames, split **by whole match with val/test pinned** to the
same matches as the previous ship so all rows below are directly comparable.
Raw/processed data and virtual environments are **not** shipped.

Environment: Python 3.12.10, torch 2.6.0+cu124 (NVIDIA RTX 4060), numpy 2.5.3.

Commands:

```
python training/build_dataset.py --raw training/datasets/raw \
    --out training/datasets/processed_v3 --version teacher-v3 --teacher smart
python training/build_dataset.py --raw training/datasets/raw training/datasets/raw_dagger \
    --out training/datasets/processed_dagger2 --version teacher-v3-dagger2 --teacher smart \
    --fixed-splits training/datasets/processed_v3/splits.json
python training/train.py --data training/datasets/processed_dagger2 \
    --out training/runs --epochs 50 --chunk 120 \
    --class-weight-power 0.35 --grad-clip 1.0
python tools/kit.py seal \
    --checkpoint training/runs/20261011_025357/best.pt \
    --provenance training/runs/20261011_025357/provenance.json
```

## Class weighting (why power 0.35)

The earlier `teacher-v2` run used full inverse-frequency class weights
(Jump 30.7×, Front dodge 9.5×). That over-predicted Jump (~4.8×) and made the bot
hop constantly in the live GUI, which forced a post-hoc calibration
(`recalibrate.py`, α=0.35) to undo the bias in `head.bias`.

`teacher-v3` instead tempers the weights **at the source**: each weight is
`(N / (4·counts))^0.5` (Jump ≈ 5.1×, Front dodge ≈ 2.8×), with gradient clipping
for stability — no recalibration needed.

For the DAgger stage the weighting was reduced further to **power 0.35**, because
the on-policy data made the model over-predict flips (Dodge precision dropped)
at power 0.5. Power 0.35 lands the predicted flip rates almost exactly on the
teacher's true rates (Jump 630 vs 643 on test) while improving accuracy on both
rare classes.

## Measured offline results (teacher-forced)

Validation split (97,522 frames; true counts: Neutral 28,376, Chase 65,960,
Jump 748, Front dodge 2,438):

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Neutral | 28,376 | 0.943 | 0.915 | 0.929 |
| Chase | 65,960 | 0.963 | 0.974 | 0.968 |
| Jump | 748 | 0.529 | 0.596 | 0.561 |
| Front dodge | 2,438 | 0.812 | 0.815 | 0.813 |

Predicted-action counts on val: Neutral 27,534, Chase 66,699, Jump **843**,
Front dodge 2,446; steering MAE (Chase) 0.153; flip hit-rate 0.873
(Jump 0.832, Front dodge 0.917; 197 segments).

Test split (97,081 frames; true Jump 643, Front dodge 2,119): predicted Jump
**630**, Front dodge 1,990; steering MAE 0.144; flip hit-rate **0.767**
(Jump 0.633, Front dodge 0.907; 176 segments). Per-class test precision/recall:

| Class | Support | Precision | Recall |
|---|---|---|---|
| Neutral | 37,604 | 0.969 | 0.938 |
| Chase | 56,715 | 0.953 | 0.976 |
| Jump | 643 | 0.500 | 0.490 |
| Front dodge | 2,119 | 0.805 | 0.756 |

Interpretation: the model predicts Jump at approximately the teacher's rate
(843 vs 748 on val; 630 vs 643 on test) with much higher Jump precision than the
`teacher-v2` base, and stronger Front-dodge recall. Aggregate accuracy was never
the target — per-class behaviour is. Full confusion matrices are in
`submission.json` → `training_provenance`.

## Live GUI testing

- The shipped checkpoint (`teacher-v3-dagger2`) was live-tested under the RLBot
  GUI in real 1v1 matches. **Better:** random jumping and flipping is clearly
  reduced and flip timing is saner. It beats Beginner and plays competitively,
  but still loses to Pro. Residual artifacts from the earlier ships (occasional
  own-goal, covariate drift off the validation distribution) are much rarer but
  not fully gone.
- Earlier ships: `teacher-v2` (pre-calibration) spawned and drove but hopped
  excessively and did not front-flip; the un-augmented `teacher-v3` played worse
  than the teacher live (hopped at the start, almost never front-flipped), which
  motivated the DAgger stage. Offline metrics are not closed-loop proof.

