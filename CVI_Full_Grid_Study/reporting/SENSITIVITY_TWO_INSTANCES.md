# Original-format two-instance sensitivity figure

`plot_sensitivity_two_instances.py` reconstructs the style of the user's
`/Users/qiluo/Downloads/sensitivity_residual.pdf` without reading or reusing its
curve values. The confirmed updated panels are (a) n=1000,m=1 and (b) n=100,m=100.

The recovered PDF format is 792×345.6 points (11×4.8 inches), STIXGeneral/stix,
11-point axes/ticks, 11.5-point panel titles, and a 10.3-point seven-entry legend
at the original right-hand coordinates. Axes bounds, xlabel/title positions and
legend text baselines match the reference PDF coordinates. Gray levels, 1.65-point
method strokes, square/circle/triangle/diamond markers, dash patterns, 10% bands,
15% legend patch, grids, tick lengths and the dashed target line are reconstructed
from PDF drawing operators. The separately edited reference ylabel's baseline x
is also matched. No historical numerical curve coordinates enter the new plot.

Production input consists of two explicit selected settings, each with ten
completed seeds, five methods and 1000-second caps. Each needs a passed full
factorwise structural audit covering the selected cases and binding every
manifest/run/point/diagnostic file. A complete larger source dataset is allowed;
a selected copied snapshot may supply its own passed audit when its case files
are byte-identical. Preflight/incomplete audits, wrong source/instance hashes,
missing methods/seeds, and unmeasured requested settings are refused before an
output folder is created. There is no fallback to n=50 or another m row.

From the workspace:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/plot_sensitivity_two_instances.py plot \
  --left-results CVI_Full_Grid_Study/results_single/verified \
  --left-n 1000 --left-m 1 \
  --right-results CVI_Full_Grid_Study/results_n100/verified \
  --right-n 100 --right-m 100 \
  --right-audit /absolute/path/to/passed/selected_MC/structural_audit.json \
  --out CVI_Updated_Sensitivity_Figure
```

The left audit defaults to the completed single-release structural audit. The
right audit can cover the prioritized ten n100/m100 cases before other MC rows
finish: independently audit the exported ten-case snapshot with the existing
structural auditor's declared-plan scope, then pass that audit file explicitly.
The plotter checks selected snapshot file hashes against the actual chosen
measurement root, so copying is not a substitute for numerical/record checks.
`--left-audit`, `--right-audit`, dimensions and roots are explicit flexible options;
defaults always use the confirmed n100 right panel.

Outputs are `sensitivity_residual.pdf/.png`, `sensitivity_residual_caption.tex`,
interpolated plot-data CSV, and hash-bound provenance. The output must be new and
outside source data/code and the delivered single paper. No worker/controller,
solver, method, source dataset, delivered article or existing ZIP is modified.

Curves are medians over a fixed complete ten-seed set. The common logarithmic
display grid also retains observed knots; each method ends at the earliest
terminal checkpoint among its ten seeds, with no extrapolation or monotone fit.
Pointwise 95% percentile-bootstrap intervals use 4000 replicates and seed 20261029
by default. Display interpolation and the 1e-12 floor do not change table values
or saved-point threshold decisions. These numerical gaps are not formal upper
certificates, and the plotter performs no independent gap solve.

Recorded algorithm/diagnostic exceptions require explicit `--allow-exceptions`.
An invalid/missing seed trajectory makes that whole method aggregate unavailable;
it never silently reduces the ten-seed denominator. Such groups and raw method
termination exceptions are visibly annotated and enumerated in provenance.

Style-only review uses explicitly synthetic data and a conspicuous fixture
banner, a different output filename, and a temporary output directory:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/plot_sensitivity_two_instances.py fixture-preview \
  --out /private/tmp/cvi_sensitivity_style_preview
```

That command cannot create the production figure filename. Its output is not an
updated experimental result. `test_sensitivity_two_instances.py` uses synthetic
fixtures only for geometry, fixed-seed aggregation, missing-input refusal and
copied-audit hash checks.
