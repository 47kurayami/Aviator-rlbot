"""Phase 3 DAgger automated match runner.

Starts matches through the already-running RLBot server (no GUI clicking) using
`rlbot.managers.MatchManager`, with the car lineup from training/dagger/match.toml:

  * car 1 = the DAgger recorder (training/dagger/bot.toml -> training/record_dagger.py).
    The STUDENT (model/V13.pt) drives; the scripted teacher supplies the label.
  * car 2 = a built-in Psyonix AllStar opponent.

Every match's rollout is appended to training/datasets/raw_dagger/ (one JSONL per
match). Each new StartMatch re-launches the recorder, so N matches -> N files.

Requires Rocket League + RLBotServer to be running. Safe with the GUI open: it
reuses the existing server and never starts/kills it.

Usage:
    .\\.venv\\Scripts\\python.exe -u training\\run_dagger_matches.py --matches 8
"""
import argparse
import time
from pathlib import Path

from rlbot import flat
from rlbot.managers import MatchManager

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "training" / "dagger" / "match.toml"
RAW_DAGGER = ROOT / "training" / "datasets" / "raw_dagger"
LIVE_PHASES = None  # filled below


def count_files():
    return len(list(RAW_DAGGER.glob("*.jsonl"))) if RAW_DAGGER.exists() else 0


def newest_file():
    files = list(RAW_DAGGER.glob("*.jsonl")) if RAW_DAGGER.exists() else []
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def is_live(packet):
    return packet is not None and packet.match_info.match_phase not in (
        flat.MatchPhase.Inactive,
        flat.MatchPhase.Ended,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--per-match-seconds", type=float, default=360.0,
                    help="max time to wait for a match to end before restarting")
    ap.add_argument("--start-timeout", type=float, default=90.0,
                    help="max time to wait for a freshly started match to go live")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--idle-stop-seconds", type=float, default=30.0,
                    help="match is considered over when the recorder's file stops growing this long")
    ap.add_argument("--dry-run", action="store_true",
                    help="only parse the config and print the plan; touch nothing")
    args = ap.parse_args()

    print(f"config     : {args.config}")
    print(f"matches    : {args.matches}")
    print(f"raw_dagger : {RAW_DAGGER}  (exists={RAW_DAGGER.exists()}, "
          f"files={count_files()})", flush=True)

    if args.dry_run:
        from rlbot.config import load_match_config
        cfg = load_match_config(args.config)
        print(f"[dry-run] parsed OK: players={len(cfg.player_configurations)} "
              f"game_mode={cfg.game_mode} auto_start={cfg.auto_start_agents}")
        return

    mgr = MatchManager()
    mgr.ensure_server_started()
    print(f"[server] running, port={mgr.rlbot_server_port}", flush=True)

    for i in range(1, args.matches + 1):
        before = count_files()
        print(f"[match {i}/{args.matches}] sending StartMatch...", flush=True)
        mgr.start_match(args.config, wait_for_start=False)

        t0 = time.time()
        while time.time() - t0 < args.start_timeout:
            if is_live(mgr.packet):
                break
            time.sleep(0.5)
        else:
            print(f"[match {i}] WARNING: never went live in {args.start_timeout:.0f}s "
                  f"(phase={getattr(mgr.packet.match_info, 'match_phase', None) if mgr.packet else None})",
                  flush=True)
        print(f"[match {i}] live; waiting up to {args.per_match_seconds:.0f}s for end", flush=True)

        t1 = time.time()
        last_size = -1
        last_growth = time.time()
        while time.time() - t1 < args.per_match_seconds:
            p = mgr.packet
            if p is not None and p.match_info.match_phase == flat.MatchPhase.Ended:
                break
            if count_files() > before:
                newest = newest_file()
                size = newest.stat().st_size if newest else -1
                if size != last_size:
                    last_size = size
                    last_growth = time.time()
                if time.time() - last_growth > args.idle_stop_seconds:
                    break
            time.sleep(0.5)
        print(f"[match {i}] finished ({time.time() - t1:.0f}s); "
              f"raw_dagger files {before} -> {count_files()}", flush=True)
        time.sleep(args.settle)

    print("[done] stopping match", flush=True)
    try:
        mgr.stop_match()
    finally:
        mgr.disconnect()
    print(f"[done] total raw_dagger files: {count_files()}", flush=True)


if __name__ == "__main__":
    main()
