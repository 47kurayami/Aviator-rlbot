"""Bake a class-prior correction into a trained checkpoint's final bias.

train.py optimises a class-weighted cross-entropy. With weights w_k the network
learns logits ~ log(w_k) + log p(k | x), so argmax over-predicts the rare
up-weighted classes. The sealed run weighted Jump ~31x and Front dodge ~9x, and
live testing showed exactly that: the bot hopped constantly but never flipped
(the flip also needs a rare 3-frame sequence imitation cannot reproduce).

Subtracting alpha*log(w_k) from the four mode logits recovers the unweighted
decision rule. It is folded into the existing ``head.bias`` tensor, so the
architecture, tensor shapes and the 26,181 parameter count are untouched -- only
bias values change. alpha=1.0 removes the weighting bias completely; lower it
(e.g. 0.5) to keep a little of the rare-class eagerness.

Runs anywhere torch runs (macOS MPS, Windows CPU/CUDA, Linux). Device only
affects speed; the saved checkpoint is always CPU tensors.

Usage:
    python training/recalibrate.py \
        --checkpoint training/runs/<ts>/best.pt \
        --data training/datasets/processed \
        --out training/runs/<ts>/best_calibrated.pt --alpha 1.0
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))
from policy import Policy  # noqa: E402
import train as T  # noqa: E402

MODE_NAMES = ("Neutral", "Chase", "Jump", "Front dodge")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_class_counts(data, ids):
    counts = np.zeros(4, dtype=np.float64)
    for mid in ids:
        d = np.load(data / f"{mid}.npz")
        for k in range(4):
            counts[k] += int((d["mode"] == k).sum())
    return counts


def evaluate(model, matches, device, chunk, steer_weight):
    if not matches:
        return None
    return T.evaluate(model.eval().to(device), matches, device, chunk, steer_weight)


def predicted_counts(metrics):
    conf = np.asarray(metrics["confusion"], dtype=np.int64)
    pred = conf.sum(axis=0)
    total = int(pred.sum())
    return {MODE_NAMES[k]: int(pred[k]) for k in range(4)}, total


def _fmt(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def summary_line(tag, metrics):
    names = {k: metrics[k] for k in MODE_NAMES}
    return (f"  {tag:<8} pred_count={predicted_counts(metrics)[0]} "
            f"steer_mae={_fmt(metrics['steer_mae_chase'], '.4f')} "
            f"jump_prec={_fmt(names['Jump']['precision'])} jump_rec={_fmt(names['Jump']['recall'])} "
            f"flip_hit={_fmt(metrics['flip']['hit_rate'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="trained best.pt to correct")
    ap.add_argument("--data", type=Path, default=ROOT / "training/datasets/processed")
    ap.add_argument("--out", type=Path, required=True, help="where to write the corrected checkpoint")
    ap.add_argument("--provenance", type=Path, default=None,
                    help="provenance JSON path (default: provenance_calibrated.json next to --out)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--steer-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))

    counts = train_class_counts(args.data, splits["train"])
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (4.0 * counts)
    log_w = np.log(weights)
    shift = -args.alpha * log_w
    print(f"device={device} alpha={args.alpha}")
    print("class_counts_train:", {MODE_NAMES[k]: int(counts[k]) for k in range(4)})
    print("class_weights    :", {MODE_NAMES[k]: round(float(weights[k]), 4) for k in range(4)})
    print("bias_shift(logit):", {MODE_NAMES[k]: round(float(shift[k]), 4) for k in range(4)})

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert isinstance(saved, dict) and "state_dict" in saved, "checkpoint needs a state_dict"
    state = {k: v.detach().clone() for k, v in saved["state_dict"].items()}
    base = Policy().eval()
    base.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in base.parameters()) == 26181, "unexpected parameter count"

    val = T.load_split(args.data, splits["val"])
    test = T.load_split(args.data, splits["test"])
    before_val = evaluate(base, val, device, args.chunk, args.steer_weight)
    before_test = evaluate(base, test, device, args.chunk, args.steer_weight)

    # Fold the correction into the existing final-layer bias.
    assert "head.bias" in state and state["head.bias"].shape == (5,), "unexpected head bias"
    old_bias = state["head.bias"].clone()
    state["head.bias"][:4] = old_bias[:4] + torch.tensor(shift, dtype=old_bias.dtype)
    assert torch.isfinite(state["head.bias"]).all(), "non-finite bias after correction"

    corrected = Policy().eval()
    corrected.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in corrected.parameters()) == 26181, "parameter count changed"
    assert all(torch.isfinite(v).all() for v in state.values()), "non-finite weights"

    after_val = evaluate(corrected, val, device, args.chunk, args.steer_weight)
    after_test = evaluate(corrected, test, device, args.chunk, args.steer_weight)

    print("validation (before -> after):")
    print(summary_line("before", before_val))
    print(summary_line("after", after_val))
    print("test (before -> after):")
    print(summary_line("before", before_test))
    print(summary_line("after", after_test))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in state.items()}}, args.out)
    out_sha = sha256(args.out)
    print("wrote", args.out, out_sha[:16] + "...")

    starter_path = ROOT / "model/V13.pt"
    starter = torch.load(starter_path, map_location="cpu", weights_only=True)["state_dict"]

    def flat(sd):
        return torch.cat([v.detach().float().reshape(-1) for v in sd.values()])

    a, b = flat(state), flat(starter)
    cosine = float((a @ b) / (a.norm() * b.norm() + 1e-12))

    base_prov = {}
    run_prov = args.checkpoint.parent / "provenance.json"
    if run_prov.is_file():
        base_prov = json.loads(run_prov.read_text(encoding="utf-8"))
    # Keep a repo-relative path so the shipped provenance has no machine-specific path.
    dataset_manifest = args.data / "dataset_manifest.json"
    try:
        dataset_manifest = dataset_manifest.resolve().relative_to(ROOT)
    except ValueError:
        pass
    prov = dict(base_prov)
    prov.update({
        "checkpoint_sha256": out_sha,
        "dataset_manifest": dataset_manifest.as_posix(),
        "cosine_similarity_vs_starter": cosine,
        "checkpoint_selection": base_prov.get("checkpoint_selection", "best validation loss")
        + "; then a class-prior logit correction was folded into head.bias (training/recalibrate.py)",
        "objective": base_prov.get("objective", "class-weighted mode CE + steer MSE")
        + f"; at inference the class-weight bias is removed by subtracting {args.alpha}*log(w_k) "
          "from the 4 mode logits",
        "calibration": {
            "method": "class-prior logit correction baked into head.bias (weights unchanged in shape)",
            "reason": "class-weighted CE made the policy over-predict rare Jump/Front dodge; live review "
                      "showed constant hopping and no flips",
            "alpha": args.alpha,
            "class_counts_train": {MODE_NAMES[k]: int(counts[k]) for k in range(4)},
            "class_weights": {MODE_NAMES[k]: float(weights[k]) for k in range(4)},
            "mode_logit_shift": {MODE_NAMES[k]: float(shift[k]) for k in range(4)},
            "source_checkpoint_sha256": sha256(args.checkpoint),
        },
        "validation_results_before": before_val,
        "validation_results_after": after_val,
        "test_results_before": before_test,
        "test_results_after": after_test,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": device},
        "limitations": base_prov.get("limitations", "") +
        " | calibration changes only the 4 mode biases; it cannot create flip sequences the network "
        "never learned, so flips likely remain rare.",
    })
    prov_path = args.provenance or (args.out.parent / "provenance_calibrated.json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print("wrote", prov_path)


if __name__ == "__main__":
    main()
