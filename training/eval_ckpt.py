"""Compare one or more checkpoints on one dataset split, apples-to-apples.

Usage:
    python training/eval_ckpt.py --data training/datasets/processed_v3 \
        --split val --ckpt A.pt B.pt ...

Prints, per checkpoint: a 4x4 confusion matrix, true vs predicted counts per
mode, per-mode precision/recall, and cosine similarity against the untouched
starter weights (the CI anti-noise gate). This is the tool used to pick a winner
between training runs on identical frames; train.py only scores the run it just
produced.

Teacher-forced (the splits hold scripted-teacher states), so it ranks checkpoints
fairly against each other but does NOT prove closed-loop gameplay.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from policy import Policy  # noqa: E402

MODE = ("Neutral", "Chase", "Jump", "Front dodge")


def load_split(data, ids):
    out = []
    for mid in ids:
        z = np.load(data / f"{mid}.npz")
        out.append({k: z[k] for k in ("obs", "mode", "steer")})
    return out


@torch.no_grad()
def evaluate(model, matches, device="cpu", chunk=120):
    model.eval()
    conf = np.zeros((4, 4), int)
    for m in matches:
        obs, mode = m["obs"], m["mode"]
        hidden = None
        preds = []
        for s in range(0, len(obs), chunk):
            e = min(s + chunk, len(obs))
            x = torch.from_numpy(obs[s:e]).to(device)
            logits, _, hidden = model(x, hidden)
            preds.append(logits.argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
        for k in range(4):
            conf[k] += np.bincount(pred[mode == k], minlength=4)[:4]
    return conf


def report(name, conf):
    print(f"\n== {name} ==")
    print("            " + "".join(f"{'pred ' + MODE[j]:>14}" for j in range(4)))
    for k in range(4):
        print(f"true {MODE[k]:>11}: " + "".join(f"{conf[k, j]:>14}" for j in range(4)))
    true = conf.sum(1)
    pred = conf.sum(0)
    print("true counts :", {MODE[k]: int(true[k]) for k in range(4)})
    print("pred counts :", {MODE[k]: int(pred[k]) for k in range(4)})
    for k in range(4):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {MODE[k]:>11}: precision={prec:.3f} recall={rec:.3f} support={int(true[k])}")


def cosine_vs(sd, starter):
    a = torch.cat([sd[k].detach().float().reshape(-1) for k in sd])
    b = torch.cat([starter[k].detach().float().reshape(-1) for k in starter])
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/datasets/processed_v3")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--starter", default="training/old_V13_starter.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data = Path(args.data)
    splits = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    matches = load_split(data, splits[args.split])
    print(f"data={data} split={args.split} matches={len(matches)} "
          f"frames={sum(len(m['mode']) for m in matches)}")

    starter = torch.load(args.starter, map_location="cpu", weights_only=True)["state_dict"]
    for c in args.ckpt:
        path = Path(c)
        if not path.exists():
            print(f"\n== {path.name} == MISSING ({path})")
            continue
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        sd = loaded["state_dict"] if "state_dict" in loaded else loaded
        model = Policy()
        model.load_state_dict(sd, strict=True)
        model.to(args.device)
        report(path.name, evaluate(model, matches, args.device))
        print("cosine vs starter:", round(cosine_vs(sd, starter), 5))


if __name__ == "__main__":
    main()
