"""Turn raw teacher recordings into labeled tensors and whole-match splits.

Reads training/datasets/raw/*.jsonl (one match per file), converts the teacher's
raw 8-channel controls into the shipped (mode, steer) label space using the same
rules as src/runtime.decode, writes one .npz per match, and assigns whole-match
train/val/test splits with a fixed seed.

The label mapping (reverse of CHANNELS = throttle, steer, pitch, yaw, roll, jump,
boost, handbrake) is:

    jump and pitch == -1  -> mode 3 (Front dodge), steer 0
    jump                  -> mode 2 (Jump),        steer 0
    throttle == 1         -> mode 1 (Chase),       steer = teacher steer
    otherwise             -> mode 0 (Neutral),     steer 0
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODE_NAMES = ("Neutral", "Chase", "Jump", "Front dodge")


def label(controls):
    throttle = float(controls[0])
    steer = float(controls[1])
    pitch = float(controls[2])
    jump = bool(controls[5])
    if jump and pitch <= -0.999:
        return 3, 0.0
    if jump:
        return 2, 0.0
    if throttle >= 0.999:
        return 1, float(np.clip(steer, -1.0, 1.0))
    return 0, 0.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_match(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path.name}: empty recording")
    rows.sort(key=lambda r: r["callback"])
    obs = np.asarray([r["obs"] for r in rows], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 13:
        raise ValueError(f"{path.name}: expected [T,13] observations, got {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError(f"{path.name}: non-finite observation")
    modes = np.empty(len(rows), dtype=np.int64)
    steer = np.empty(len(rows), dtype=np.float32)
    elapsed = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        m, s = label(r["controls"])
        modes[i] = m
        steer[i] = s
        elapsed[i] = r["elapsed"]
    return {"obs": obs, "mode": modes, "steer": steer, "elapsed": elapsed,
            "callback": np.asarray([r["callback"] for r in rows], dtype=np.int64),
            "match_id": rows[0]["match_id"], "index": rows[0].get("index"),
            "team": rows[0].get("team")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, nargs="+", default=[Path("training/datasets/raw")],
                        help="one or more raw recording dirs (later dirs can hold DAgger data)")
    parser.add_argument("--out", type=Path, default=Path("training/datasets/processed"))
    parser.add_argument("--fixed-splits", type=Path, default=None,
                        help="reuse an existing splits.json; any match not listed there is "
                             "added to TRAIN (keeps val/test identical across rounds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--teacher", default="reference")
    parser.add_argument("--version", default="scripted-teacher-v1")
    args = parser.parse_args()

    raw_files = sorted(p for directory in args.raw for p in directory.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit(f"no raw recordings in {[str(d) for d in args.raw]}")
    args.out.mkdir(parents=True, exist_ok=True)

    matches = {}
    for path in raw_files:
        match = load_match(path)
        if len(match["obs"]) < args.min_length:
            print(f"skip {path.name}: {len(match['obs'])} callbacks")
            continue
        matches[match["match_id"]] = (match, path)
    if not matches:
        raise SystemExit("no usable recordings")

    ids = sorted(matches)
    if args.fixed_splits:
        fixed = json.loads(args.fixed_splits.read_text(encoding="utf-8"))
        split_ids = {k: list(fixed[k]) for k in ("train", "val", "test")}
        assigned = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
        missing = sorted(i for i in assigned if i not in matches)
        if missing:
            raise SystemExit(f"fixed splits reference {len(missing)} missing matches, e.g. {missing[:3]}")
        extra = [i for i in ids if i not in assigned]
        split_ids["train"].extend(extra)
        print(f"fixed splits: {len(fixed['train'])} train + {len(extra)} new -> train")
    else:
        order = np.random.default_rng(args.seed).permutation(len(ids))
        n = len(ids)
        if n >= 3:
            n_val = max(1, int(round(n * args.val_frac)))
            n_test = max(1, int(round(n * (1.0 - args.train_frac - args.val_frac))))
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = 1, 0, 0
        split_ids = {"train": [], "val": [], "test": []}
        for rank, idx in enumerate(order):
            if rank < n_train:
                split_ids["train"].append(ids[idx])
            elif rank < n_train + n_val:
                split_ids["val"].append(ids[idx])
            else:
                split_ids["test"].append(ids[idx])

    raw_hashes = {p.name: sha256(p) for p in raw_files}
    processed_hashes = {}
    for mid, (match, _path) in matches.items():
        target = args.out / f"{mid}.npz"
        np.savez(target, obs=match["obs"], mode=match["mode"], steer=match["steer"],
                 elapsed=match["elapsed"], callback=match["callback"])
        processed_hashes[target.name] = sha256(target)

    (args.out / "splits.json").write_text(
        json.dumps({"seed": args.seed, **split_ids}, indent=2) + "\n", encoding="utf-8")

    def stats(ids_):
        modes = np.concatenate([matches[i][0]["mode"] for i in ids_]) if ids_ else np.empty(0, int)
        return {"matches": sorted(ids_), "frames": int(len(modes)),
                "mode_counts": {MODE_NAMES[k]: int((modes == k).sum()) for k in range(4)}}

    manifest = {"version": args.version, "teacher": args.teacher,
                "created": datetime.now(timezone.utc).isoformat(),
                "split_method": "whole matches, shuffled with fixed seed",
                "seed": args.seed,
                "splits": {k: stats(v) for k, v in split_ids.items()},
                "raw_files": raw_hashes, "processed_files": processed_hashes}
    (args.out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for k, v in split_ids.items():
        print(f"{k}: {len(v)} matches, {manifest['splits'][k]['frames']} frames, "
              f"{manifest['splits'][k]['mode_counts']}")
    print("dataset:", args.out)


if __name__ == "__main__":
    main()
