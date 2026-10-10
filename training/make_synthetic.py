"""Generate synthetic raw recordings so the Phase 0 pipeline can be smoke-tested
without a game.

The output has the same JSONL schema as record_teacher.py, with every mode
represented, so build_dataset.py and train.py can be exercised end to end.
"""
import argparse
import json
import math
from pathlib import Path


def controls_for(step):
    cycle = step % 60
    if cycle < 4:
        return [0.0] * 8                                   # Neutral
    if cycle < 10:
        return [0.0, 0.0, 0.0, 0.0, 0.0, True, False, False]   # Jump
    if cycle < 15:
        return [0.0, 0.0, -1.0, 0.0, 0.0, True, False, False]  # Front dodge
    return [1.0, math.sin(step * 0.1), 0.0, 0.0, 0.0, False, False, False]  # Chase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("training/datasets/raw"))
    parser.add_argument("--matches", type=int, default=6)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = __import__("random").Random(args.seed)
    for m in range(args.matches):
        match_id = f"synthetic_{m:03d}"
        elapsed = 0.0
        with (args.out / f"{match_id}.jsonl").open("w", encoding="utf-8") as f:
            for step in range(1, args.frames + 1):
                elapsed += 1.0 / 60.0
                obs = [round(rng.uniform(-1.5, 1.5), 6) for _ in range(13)]
                obs[8] = 1.0
                obs[9] = 1.0
                f.write(json.dumps(dict(v=1, match_id=match_id, callback=step, frame=step * 2,
                                        elapsed=elapsed, index=0, team=0,
                                        obs=obs, controls=controls_for(step))) + "\n")
    print(f"synthetic raw: {args.matches} matches x {args.frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
