"""Summarize raw checkpoints and draw publication figures (run after both drivers)."""

import argparse
from io import BytesIO
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(raw):
    records = []
    for (n, m, method, seed), data in raw.groupby(["n", "m", "method", "seed"]):
        data = data.sort_values("k")
        hit = data[(data.residual <= data.target.iloc[0]) &
                   (data.gap_status == "optimal")]
        first = hit.iloc[0] if len(hit) else None
        last = data.iloc[-1]
        records.append(dict(n=n, m=m, method=method, seed=seed,
                            reached=first is not None,
                            time_to_target=first.elapsed if first is not None else np.nan,
                            F_to_target=first.F_calls if first is not None else np.nan,
                            grad_to_target=first.grad_calls if first is not None else np.nan,
                            component_grad_to_target=m * first.grad_calls if first is not None and method == "MC-PFAOS" else np.nan,
                            proj_to_target=first.projection_calls if first is not None else np.nan,
                            final_residual=last.residual, final_gap=last.gap,
                            final_violation=last.violation,
                            final_elapsed=last.elapsed,
                            final_projection_time=last.projection_time,
                            final_status=last.solver_status,
                            multiplier_max=last.multiplier_max,
                            gap_status=last.gap_status))
    return pd.DataFrame.from_records(records)


def _number(v, digits=2):
    if pd.isna(v):
        return "--"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}g}"


def tex_table(summary, settings, output, multi, target, axis=None):
    unit = axis or ("m" if multi else "n")
    method = "MC-PFAOS" if multi else "PFAOS"
    label = ("tab:mc_dimension_numerical" if multi and unit == "n" else
             "tab:mc_pfaos_numerical" if multi else "tab:pfaos_numerical")
    title = ("MC-PFAOS versus projected OE, varying dimension at $m=5$" if multi and unit == "n" else
             "MC-PFAOS versus projected OE, varying the number of constraints" if multi else
             "PFAOS versus projected OE, varying dimension")
    caption = (rf"\caption{{{title}. Ten seeds per setting; target $\varepsilon={target:g}$. "
               "Time and work are medians over successful runs; final residuals use all seeds. "
               "FO/proj counts full Jacobian calls" if multi else
               rf"\caption{{{title}. Ten seeds per setting; target $\varepsilon={target:g}$. "
               "Time and work are medians over successful runs; final residuals use all seeds. "
               "FO/proj counts constraint-gradient calls")
    caption += (" and $mN_J$ counts individual component-gradient equivalents" if multi else "")
    caption += " for the proposed method, or generic projections for OE; dashes indicate no target crossing.}"
    lines = [r"\begin{table}[!htbp]", r"\centering\scriptsize",
             caption,
             rf"\label{{{label}}}",
             r"\begin{tabular}{r l r r r r r r}" if multi else r"\begin{tabular}{r l r r r r r}",
             r"\hline", rf"${unit}$ & Method & Reached & Time (s) & $F$ calls & FO/proj & "
             + (r"$mN_J$ & " if multi else "") + r"Final $\mathcal E$ \\",
             r"\hline"]
    for setting in settings:
        for name in [method, "Projected OE"]:
            sub = summary[(summary[unit] == setting) & (summary.method == name)]
            assert len(sub) > 0, (setting, name)
            hit = sub[sub.reached]
            work = "grad_to_target" if name == method else "proj_to_target"
            extra = (_number(hit.component_grad_to_target.median(), 3) + " & "
                     if multi and name == method else "-- & " if multi else "")
            lines.append(f"{setting} & {name} & {len(hit)}/{len(sub)} & "
                         f"{_number(hit.time_to_target.median(), 3)} & "
                         f"{_number(hit.F_to_target.median(), 3)} & "
                         f"{_number(hit[work].median(), 3)} & "
                         + extra + f"{_number(sub.final_residual.median(), 3)}" + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table}"])
    Path(output).write_text("\n".join(lines) + "\n")


def time_profile(data, times, metric):
    """Median/IQR of the latest *saved* checkpoint available at each time."""
    paths = []
    for _, run in data.groupby("seed"):
        run = run.sort_values("k")
        assert run.k.iloc[0] == 0
        elapsed = run.elapsed.to_numpy().copy()
        elapsed[0] = 0.0  # Both methods have the same supplied starting point.
        assert np.all(np.diff(elapsed) >= 0)
        index = np.searchsorted(elapsed, times, side="right") - 1
        assert np.min(index) >= 0
        paths.append(run[metric].to_numpy()[index])
    paths = np.stack(paths)
    assert np.isfinite(paths).all()
    return np.percentile(paths, [25, 50, 75], axis=0)


def _time_axis(ax, xmax=3.0):
    ax.set_xscale("linear")
    ax.set_xlim(0, xmax)
    ticks = {
        3.0: [0, .5, 1, 1.5, 2, 2.5, 3],
        1.2: [0, .2, .4, .6, .8, 1, 1.2],
        .12: [0, .04, .08, .12],
        .02: [0, .005, .01, .015, .02],
    }
    ax.set_xticks(ticks[xmax])
    ax.set_xlabel("Wall-clock time (s)")
    ax.grid(alpha=0.17)


def _plot_metric(ax, data, proposed, metric, target, xmax=3.0):
    shown = data[(data.k > 0) & (data.elapsed <= xmax)].elapsed.to_numpy()
    times = np.unique(np.r_[0, shown, xmax])
    for method, color in [(proposed, "#1767a0"), ("Projected OE", "#c05621")]:
        low, middle, high = time_profile(data[data.method == method], times, metric)
        ax.step(times, middle, where="post", lw=1.8, color=color, label=method)
        ax.fill_between(times, low, high, step="post", color=color, alpha=0.13)
    ax.scatter([0], [data[data.k == 0][metric].median()], s=16, c="#333333", zorder=5)
    ax.axhline(target, ls="--", lw=1, color="#777777")
    _time_axis(ax, xmax)
    if metric == "gap":
        # symlog shows the exact zero gap when an iterate overshoots the
        # feasible solution; an ordinary log scale cannot represent zero.
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"Weak VI gap $\mathrm{Gap}(\bar x)$")
    else:
        ax.set_ylabel(r"Constraint violation $\|[G(\bar x)]_+\|_2$")
        ax.set_ylim(bottom=0)


def _save(fig, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = BytesIO()
    fig.savefig(pdf, format="pdf")
    data = pdf.getvalue()
    assert data.rstrip().endswith(b"%%EOF")
    with out.with_suffix(".pdf").open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    fig.savefig(out.with_suffix(".png"), dpi=240)
    plt.close(fig)


def figures(single, multi, output, target):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    # Every target crossing in the n grid occurs by 1.2 seconds. The six
    # panels share the same linear time axis so target times align directly.
    chosen_n = [25, 50, 100, 200, 500, 1000]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes.flat, chosen_n):
        data = single[single.n == n]
        _plot_metric(ax, data, "PFAOS", "gap", target, xmax=1.2)
        ax.set_title(rf"$n={n}$")
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, output / "pfaos_residual")

    fig, ax = plt.subplots(figsize=(6.6, 2.65), constrained_layout=True)
    # All six dimensions and both methods have zero violation in the raw data.
    if (single.violation == 0).all():
        ax.plot([0, 1.2], [0, 0], lw=2, color="#303f51")
        ax.scatter([0], [0], s=17, c="#303f51", zorder=5)
        ax.text(0.48, 0.76, "PFAOS and projected OE: zero at every saved checkpoint\nfor all six dimensions",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.axhline(target, ls="--", lw=1, color="#777777")
        _time_axis(ax, xmax=1.2)
        ax.set_ylim(0, target * 1.6)
        ax.set_ylabel(r"Constraint violation $\|[G(\bar x)]_+\|_2$")
    else:
        for n in chosen_n:
            _plot_metric(ax, single[single.n == n], "PFAOS", "violation", target)
    _save(fig, output / "pfaos_violation")

    data = multi[(multi.n == 50) & (multi.m == 100)]
    for metric in ["gap", "violation"]:
        early = .02 if metric == "gap" else .12
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True,
                                 constrained_layout=True)
        for ax, horizon in zip(axes, (early, 3.0)):
            _plot_metric(ax, data, "MC-PFAOS", metric, target, xmax=horizon)
            ax.set_title(f"First {horizon:g} s" if horizon < 3 else "Full 3 s")
        axes[1].set_ylabel("")
        axes[1].legend(frameon=False, fontsize=9,
                       loc="lower left" if metric == "gap" else "upper right")
        _save(fig, output / f"mc_{'residual' if metric == 'gap' else metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfaos", type=Path, required=True)
    parser.add_argument("--mc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    single = pd.read_csv(args.pfaos)
    multi = pd.read_csv(args.mc)
    single_summary, multi_summary = summarize(single), summarize(multi)
    single_summary.to_csv(args.output / "pfaos_runs.csv", index=False)
    multi_summary.to_csv(args.output / "mc_runs.csv", index=False)
    tex_table(single_summary, sorted(single.n.unique()),
              args.output / "pfaos_table.tex", False, args.target)
    fixed_n = 50
    tex_table(multi_summary[multi_summary.n == fixed_n],
              sorted(multi[multi.n == fixed_n].m.unique()),
              args.output / "mc_table.tex", True, args.target)
    dimension = multi_summary[multi_summary.m == 5]
    tex_table(dimension, sorted(dimension.n.unique()),
              args.output / "mc_dimension_table.tex", True, args.target, axis="n")
    for old in ("pfaos_convergence", "mc_convergence"):
        for suffix in (".pdf", ".png"):
            (args.output / f"{old}{suffix}").unlink(missing_ok=True)
    figures(single, multi, args.output, args.target)
    print("Written figures, summaries, and LaTeX tables to", args.output)


if __name__ == "__main__":
    main()
