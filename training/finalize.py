"""Verify a saved checkpoint and (re)write its provenance.json.

train.py only writes provenance.json/metrics.json at the very end of a full run,
so an early stop leaves a usable best.pt but no provenance. This tool reproduces
the seal-critical checks (strict load, 26181 params, finite weights, hash/cosine
vs the starter) and evaluates best.pt on the val/test splits to fill an honest
provenance.json that `tools/kit.py seal` accepts.

Usage:
    python training/finalize.py --run training/runs/<timestamp>
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(state):
    return torch.cat([v.detach().float().reshape(-1) for v in state.values()])


def cosine(a, b):
    a, b = flatten(a), flatten(b)
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir containing best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "training/datasets/processed")
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--steer-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-planned", type=int, default=30)
    ap.add_argument("--epochs-ran", type=int, default=None, help="approx epochs completed before early stop")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    best = args.run / "best.pt"
    assert best.is_file(), f"missing {best}"

    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "dataset_manifest.json").read_text(encoding="utf-8"))
    submission = json.loads((ROOT / "submission.json").read_text(encoding="utf-8"))
    baseline = submission["baseline_checkpoint_sha256"]

    # --- seal-style verification -------------------------------------------
    saved = torch.load(best, map_location="cpu", weights_only=True)
    assert isinstance(saved, dict) and "state_dict" in saved, "checkpoint needs a state_dict"
    model = Policy().eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    nparams = sum(p.numel() for p in model.parameters())
    assert nparams == 26181, f"unexpected parameter count: {nparams}"
    assert all(torch.isfinite(v).all() for v in saved["state_dict"].values()), "non-finite weights"
    with torch.inference_mode():
        h = None
        for _ in range(3):
            logits, steer, h = model(torch.zeros((1, 13), dtype=torch.float32), h)
            assert logits.shape == (1, 4) and steer.shape == (1,) and h.shape == (1, 1, 64)
            assert all(torch.isfinite(x).all() for x in (logits, steer, h)), "non-finite output"

    best_sha = sha256(best)
    starter_sha = sha256(ROOT / "model/V13.pt")
    starter = torch.load(ROOT / "model/V13.pt", map_location="cpu", weights_only=True)["state_dict"]
    sim = cosine(saved["state_dict"], starter)
    print(f"params={nparams}  best_sha={best_sha[:16]}…  starter_sha={starter_sha[:16]}…")
    print(f"differs_from_starter_file={best_sha != starter_sha}  cosine_vs_starter={sim:.6f}")

    # --- evaluation --------------------------------------------------------
    val = T.load_split(args.data, splits["val"])
    test = T.load_split(args.data, splits["test"])
    val_m = T.evaluate(model.to(device), val, device, args.chunk, args.steer_weight) if val else None
    test_m = T.evaluate(model.to(device), test, device, args.chunk, args.steer_weight) if test else None
    if val_m:
        print("val  steer_mae=%.4f flip_hit=%.4f conf=%s" % (
            val_m["steer_mae_chase"], val_m["flip"]["hit_rate"], val_m["confusion"]))
    if test_m:
        print("test steer_mae=%.4f flip_hit=%.4f conf=%s" % (
            test_m["steer_mae_chase"], test_m["flip"]["hit_rate"], test_m["confusion"]))

    # --- provenance --------------------------------------------------------
    provenance = {
        "dataset_version": manifest.get("version"),
        "dataset_manifest": str((args.data / "dataset_manifest.json").as_posix()),
        "split_method": manifest.get("split_method"),
        "seed": args.seed,
        "architecture": "13D -> Linear(13,64)+ReLU -> GRU(64,64) -> Linear(64,5); "
                        "4 mode logits + tanh steer; 26181 parameters",
        "objective": f"mode cross-entropy (class_weight=True) + {args.steer_weight} * Chase-masked steering MSE",
        "checkpoint_selection": "lowest validation loss (mode CE + steer MSE); run stopped early, "
                                "best.pt re-verified and evaluated by training/finalize.py",
        "training_command": "python training/train.py --data training/datasets/processed "
                            "--out training/runs --epochs %d --chunk %d" % (args.epochs_planned, args.chunk),
        "epochs_planned": args.epochs_planned,
        "epochs_run_before_early_stop": args.epochs_ran,
        "tbptt_chunk": args.chunk,
        "hidden_state": "per-match recurrent stream; hidden detached between chunks; "
                        "reset at match boundaries only",
        "class_counts_train": manifest.get("class_counts", {}).get("train"),
        "dataset_matches": {"train": len(splits["train"]), "val": len(splits["val"]),
                            "test": len(splits["test"])},
        "validation_results": val_m,
        "test_results": test_m,
        "checkpoint_sha256": best_sha,
        "cosine_similarity_vs_starter": sim,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": device},
        "limitations": "offline imitation metrics only; closed-loop gameplay not verified here; "
                       "scripted teacher is imperfect and flip events are sparse; the run was "
                       "stopped before all planned epochs; mixed reference/smart teacher data.",
        "provenance_note": "generated post-hoc by training/finalize.py after an early stop",
    }
    (args.run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.run / "final_eval.json").write_text(
        json.dumps({"val": val_m, "test": test_m}, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.run / "provenance.json")


if __name__ == "__main__":
    main()
