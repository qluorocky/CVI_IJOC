"""Experiment II: MC-PFAOS versus projected OE, varying m and n."""

import argparse
from pathlib import Path

from experiment_common import append_rows, run_pair, write_config
from game_problem import make_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", default="2,5,10,20,40,100,250,500", help="Constraint grid at fixed n")
    parser.add_argument("--fixed-n", type=int, default=50)
    parser.add_argument("--n", default="25,50,100", help="Dimension grid at fixed m")
    parser.add_argument("--fixed-m", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--target", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("results/reproduction_mc"))
    args = parser.parse_args()
    write_config(args.out / "config.json", args)
    settings = [(args.fixed_n, m) for m in map(int, args.m.split(",")) if m >= 2]
    settings += [(n, args.fixed_m) for n in map(int, args.n.split(","))]
    for n, m in dict.fromkeys(settings):
        for seed in range(args.seeds):
            rows = run_pair(make_game(seed, n, m), seed, args.seconds,
                            args.target, args.iterations)
            append_rows(args.out / "checkpoints.csv", rows)
            first_end = [row for row in rows if row["method"] == "MC-PFAOS"][-1]
            print(f"MC-PFAOS n={n} m={m} seed={seed} "
                  f"end residuals={first_end['residual']:.3g},"
                  f"{rows[-1]['residual']:.3g}", flush=True)


if __name__ == "__main__":
    main()
