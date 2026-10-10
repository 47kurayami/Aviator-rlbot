"""Train the frozen 26,181-parameter V13 policy on labeled teacher recordings.

Same architecture and 13-D observation as the shipped runtime; only weights are
fit. Sequence handling: one recurrent stream per match, truncated backprop with
the hidden state detached between chunks and reset only at match boundaries.
Objective: (optionally class-weighted) mode cross-entropy + Chase-masked
steering MSE. Checkpoint selection is by validation loss.

The saved checkpoint is a plain {"state_dict": ...} mapping, exactly what
tools/kit.py seal --checkpoint expects.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from policy import Policy  # noqa: E402

MODE_NAMES = ("Neutral", "Chase", "Jump", "Front dodge")


def load_split(processed, ids):
    matches = []
    for mid in ids:
        data = np.load(processed / f"{mid}.npz")
        matches.append({k: data[k] for k in ("obs", "mode", "steer")})
    return matches


def chunks(n, size):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def class_weights(matches, device):
    counts = np.zeros(4, dtype=np.float64)
    for match in matches:
        for k in range(4):
            counts[k] += int((match["mode"] == k).sum())
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (4.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def train_epoch(model, matches, opt, device, chunk, steer_weight, ce_weight):
    model.train()
    total_ce = total_st = steps = 0.0
    for i in np.random.permutation(len(matches)):
        match = matches[i]
        obs, mode, steer = match["obs"], match["mode"], match["steer"]
        hidden = None
        for start, end in chunks(len(obs), chunk):
            x = torch.from_numpy(obs[start:end]).to(device)
            target = torch.from_numpy(mode[start:end]).to(device)
            target_steer = torch.from_numpy(steer[start:end]).to(device)
            logits, steer_pred, hidden = model(x, hidden)
            loss_ce = F.cross_entropy(logits, target, weight=ce_weight)
            mask = target == 1
            loss_st = F.mse_loss(steer_pred[mask], target_steer[mask]) if mask.any() \
                else torch.zeros((), device=device)
            loss = loss_ce + steer_weight * loss_st
            opt.zero_grad()
            loss.backward()
            opt.step()
            hidden = hidden.detach()
            total_ce += float(loss_ce)
            total_st += float(loss_st)
            steps += 1
    return {"ce": total_ce / max(steps, 1.0), "steer_mse": total_st / max(steps, 1.0)}


def confusion_metrics(conf):
    out = {}
    for k in range(4):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
        out[MODE_NAMES[k]] = {"support": int(conf[k, :].sum()),
                             "precision": precision, "recall": recall, "f1": f1}
    return out


@torch.no_grad()
def evaluate(model, matches, device, chunk, steer_weight):
    model.eval()
    conf = np.zeros((4, 4), dtype=np.int64)
    steer_errors = []
    flip = {"segments": 0, "hits": 0, "latency": [], "by_mode": {2: [0, 0], 3: [0, 0]}}
    total_ce = total_st = steps = 0.0
    for match in matches:
        obs, mode, steer = match["obs"], match["mode"], match["steer"]
        preds, steer_preds = [], []
        hidden = None
        for start, end in chunks(len(obs), chunk):
            x = torch.from_numpy(obs[start:end]).to(device)
            target = torch.from_numpy(mode[start:end]).to(device)
            target_steer = torch.from_numpy(steer[start:end]).to(device)
            logits, steer_pred, hidden = model(x, hidden)
            loss_ce = F.cross_entropy(logits, target)
            mask = target == 1
            loss_st = F.mse_loss(steer_pred[mask], target_steer[mask]) if mask.any() \
                else torch.zeros((), device=device)
            total_ce += float(loss_ce)
            total_st += float(loss_st)
            steps += 1
            preds.append(logits.argmax(1).cpu().numpy())
            steer_preds.append(steer_pred.cpu().numpy())
        pred = np.concatenate(preds)
        pred_steer = np.concatenate(steer_preds)
        for k in range(4):
            conf[k] += np.bincount(pred[mode == k], minlength=4)[:4]
        chase = mode == 1
        if chase.any():
            steer_errors.append(np.abs(pred_steer[chase] - steer[chase]))
        i = 0
        while i < len(mode):
            if mode[i] in (2, 3):
                j = i
                while j < len(mode) and mode[j] in (2, 3):
                    j += 1
                first = next((k for k in range(i, j) if pred[k] in (2, 3)), None)
                flip["segments"] += 1
                flip["by_mode"][int(mode[i])][1] += 1
                if first is not None:
                    flip["hits"] += 1
                    flip["latency"].append(first - i)
                    flip["by_mode"][int(mode[i])][0] += 1
                i = j
            else:
                i += 1
    metrics = confusion_metrics(conf)
    metrics["confusion"] = conf.tolist()
    metrics["steer_mae_chase"] = float(np.concatenate(steer_errors).mean()) if steer_errors else None
    metrics["loss"] = {"ce": total_ce / max(steps, 1.0), "steer_mse": total_st / max(steps, 1.0)}
    metrics["flip"] = {
        "segments": flip["segments"],
        "hit_rate": flip["hits"] / flip["segments"] if flip["segments"] else None,
        "mean_latency": float(np.mean(flip["latency"])) if flip["latency"] else None,
        "by_mode": {MODE_NAMES[k]: {"segments": v[1],
                                    "hit_rate": (v[0] / v[1] if v[1] else None)}
                    for k, v in flip["by_mode"].items()},
    }
    return metrics


def save_checkpoint(model, path):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("training/datasets/processed"))
    parser.add_argument("--out", type=Path, default=Path("training/runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=120, help="truncated backprop length in callbacks")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steer-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "dataset_manifest.json").read_text(encoding="utf-8"))
    train = load_split(args.data, splits["train"])
    val = load_split(args.data, splits["val"])
    test = load_split(args.data, splits["test"])
    if not train:
        raise SystemExit("empty train split")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = Policy().to(device)
    assert sum(p.numel() for p in model.parameters()) == 26181, "unexpected parameter count"
    ce_weight, counts = class_weights(train, device)
    if not args.class_weight:
        ce_weight = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run = args.out / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train_matches={len(train)} val_matches={len(val)} "
          f"test_matches={len(test)} class_counts={counts.tolist()}")
    print(f"run: {run}")

    history = []
    best_score = float("inf")
    best_val = None
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train, optimizer, device, args.chunk, args.steer_weight, ce_weight)
        vm = evaluate(model, val, device, args.chunk, args.steer_weight) if val else None
        score = (vm["loss"]["ce"] + args.steer_weight * vm["loss"]["steer_mse"]) if vm else tr["ce"]
        history.append({"epoch": epoch, "train": tr, "val": vm, "score": score})
        save_checkpoint(model, run / "last.pt")
        if score < best_score:
            best_score = score
            best_val = vm
            save_checkpoint(model, run / "best.pt")
        flip_rate = None if not vm else vm["flip"]["hit_rate"]
        print(f"epoch {epoch:3d} train_ce={tr['ce']:.4f} train_steer={tr['steer_mse']:.4f} "
              f"val_ce={vm['loss']['ce']:.4f} val_flip={flip_rate} best={best_score:.4f}"
              if vm else f"epoch {epoch:3d} train_ce={tr['ce']:.4f}")
    (run / "metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    if best_val is None:
        save_checkpoint(model, run / "best.pt")
    if (run / "best.pt").exists():
        model.load_state_dict(torch.load(run / "best.pt", weights_only=True)["state_dict"])
    test_metrics = evaluate(model, test, device, args.chunk, args.steer_weight) if test else None
    if test_metrics:
        print("test:", json.dumps({k: test_metrics[k] for k in
              ("steer_mae_chase", "flip")}, indent=2))

    provenance = {
        "dataset_version": manifest.get("version"),
        "dataset_manifest": str((args.data / "dataset_manifest.json").as_posix()),
        "split_method": manifest.get("split_method"),
        "seed": args.seed,
        "architecture": "13D -> Linear(13,64)+ReLU -> GRU(64,64) -> Linear(64,5); "
                        "4 mode logits + tanh steer; 26181 parameters",
        "objective": f"mode cross-entropy (class_weight={args.class_weight}) + "
                     f"{args.steer_weight} * Chase-masked steering MSE",
        "checkpoint_selection": "lowest validation loss (mode CE + steer MSE)",
        "training_command": " ".join(sys.argv),
        "epochs": args.epochs,
        "tbptt_chunk": args.chunk,
        "hidden_state": "per-match recurrent stream; detached between chunks; reset at match boundaries only",
        "class_counts_train": {MODE_NAMES[k]: int(counts[k]) for k in range(4)},
        "validation_results": best_val,
        "test_results": test_metrics,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": device},
        "limitations": "offline imitation metrics only; closed-loop gameplay not verified here; "
                       "scripted teacher is imperfect and flip events are sparse; validate live in the GUI.",
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("checkpoint:", run / "best.pt")
    print("provenance:", run / "provenance.json")


if __name__ == "__main__":
    main()
