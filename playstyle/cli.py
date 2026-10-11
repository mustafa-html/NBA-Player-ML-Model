"""Command line: fetch data, train, and query.

    python -m playstyle.cli fetch
    python -m playstyle.cli train [--k 10 | --k auto]
    python -m playstyle.cli similar "Luka Doncic" --season 2026 [--scope all]
"""

from __future__ import annotations

import argparse
import json
import sys

from playstyle import config
from playstyle.data import fetch_raw
from playstyle.pipeline import Engine, train


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="playstyle")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Download the latest season tables from the source dataset")
    f.add_argument("--first-season", type=int, default=config.TrainConfig().first_season)
    f.add_argument("--last-season", type=int, default=config.TrainConfig().last_season)

    t = sub.add_parser("train", help="Build features, fit clusters, write artifacts and reports")
    t.add_argument("--k", default=None, help="Number of clusters, or 'auto' to pick by silhouette score")
    t.add_argument("--min-minutes", type=int, default=None)

    s = sub.add_parser("similar", help="Players with the most similar playstyle")
    s.add_argument("player")
    s.add_argument("--season", type=int, default=None)
    s.add_argument("--scope", choices=["season", "all"], default="season")
    s.add_argument("-n", type=int, default=10)

    args = p.parse_args(argv)

    if args.cmd == "fetch":
        paths = fetch_raw(seasons=range(args.first_season, args.last_season + 1))
        print("\n".join(f"{k}: {v}" for k, v in paths.items()))
        return 0

    if args.cmd == "train":
        cfg = config.TrainConfig()
        if args.k == "auto":
            cfg.k = None
        elif args.k:
            cfg.k = int(args.k)
        if args.min_minutes:
            cfg.min_minutes = args.min_minutes
        summary = train(cfg)
        print(json.dumps({k: v for k, v in summary.items() if k != "archetypes"}, indent=2))
        for a in summary["archetypes"]:
            print(f"  {a['name']:<24} {a['size']:>5}  {', '.join(a['traits'])}")
        return 0

    engine = Engine.load()
    idx = engine.index.find(args.player, args.season)
    if idx is None:
        hint = engine.index.suggest(args.player)
        print(f"No qualifying season for '{args.player}'." + (f" Did you mean: {', '.join(hint)}?" if hint else ""))
        return 1
    prof = engine.player_profile(idx)
    print(f"{prof['player']} ({prof['season']}): {prof['archetype']} — {', '.join(prof['traits'])}")
    for _, r in engine.similar(idx, n=args.n, scope=args.scope).iterrows():
        print(f"  {r['similarity']:5.1f}%  {r['player']} ({r['season']}, {r['archetype']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
