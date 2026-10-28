"""Experiment I: PFAOS versus projected OE, varying dimension n."""

import argparse
from pathlib import Path

from experiment_common import append_rows, run_pair, write_config
from game_problem import make_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", default="25,50,100,200,500,1000", help="Comma-separated dimensions, each at least 4")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--target", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("results/reproduction_pfaos"))
    args = parser.parse_args()
    write_config(args.out / "config.json", args)
    for n in map(int, args.n.split(",")):
        for seed in range(args.seeds):
            rows = run_pair(make_game(seed, n, 1), seed, args.seconds,
                            args.target, args.iterations)
            append_rows(args.out / "checkpoints.csv", rows)
            final = {row["method"]: row["residual"] for row in rows}
            print(f"PFAOS n={n} seed={seed} "
                  + ", ".join(f"{name}={value:.3g}" for name, value in final.items()),
                  flush=True)


if __name__ == "__main__":
    main()
