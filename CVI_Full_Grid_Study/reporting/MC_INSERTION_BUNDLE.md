# Separate MC insertion bundle

`mc_insertion_bundle.py` is an untimed extraction/build helper. It does not run
an experiment, re-solve a diagnostic, resample a trajectory, or edit the delivered
single-constraint article/ZIP. No real artifact is created until root invokes it.

Its only production source is `CVI_Full_Grid_Study/results_n100/verified`.
Preparation requires the exact full eleven-setting, 110-instance/550-method
record inventory, a complete measured report, a passed full-scope independent
statistics audit, a passed full-scope factorwise structural audit, and fresh
input hashes. A full report with honestly recorded single-constraint exceptions
is permitted. The eight known original invalid single-constraint gaps do not
enter MC metrics and cannot alone block an otherwise valid MC subset.

From the workspace, once the full data/report/audits exist:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/mc_insertion_bundle.py prepare
```

The fixed separate output is `CVI_MC_Insertion_Bundle/` at workspace root.
It contains:

- `MC_results/`: exact existing MC primary/gap-only tables, exact existing
  five-panel figure PDF/PNG, portable figure fragment, MC-only extracted CSVs,
  MC-only measured discussion/protocol, and diagnostic/failure inventory.
- `data/`: only the fifty new MC cases at n=100 and m=10,50,100,200,500, with
  ten seeds and five methods (250 trajectories). No single-case records.
- `provenance/`: the original full-plan metadata, full-audit hash receipts,
  selected figure provenance, and an independent MC-only recheck of the CSV and
  two fifteen-row tables. Full single measurement tables/records are not copied.
- `mc_review.tex/.pdf`, `mc_table_figure.tex/.pdf`, `mc_insert.tex`, `build.sh`,
  `bundle_manifest.json`, README, and checksums.

Preparation normally compiles both standalone PDFs twice using pdfLaTeX, without
running any numerical code. `--no-compile` permits staging first; then run
`sh build.sh` inside the new bundle. The helper copies the existing figure and
its original bootstrap display; extraction performs no new resampling.

Actual MC algorithm failures and unresolved diagnostics are enumerated. They
require explicit `prepare --allow-unresolved` to produce a qualified staging
bundle. Counts retain all ten seeds, conditional timings retain their meaning,
unavailable aggregate groups do not drop seeds, and no time cap is imputed as
a hitting time. Source table/figure qualification is preserved.

To insert manually, copy `MC_results/` beside the article's main.tex and insert
the contents of `mc_insert.tex` at the chosen MC-results location. The figure
uses its own `MCResultsDir`, leaving the single paper's `ResultsDir` alone.
The gap-only companion can be inserted separately in the numerical supplement.
No helper operation writes into the delivered single paper or original data.

After reviewing the actual standalone PDFs, root writes
`CVI_Full_Grid_Study/notes/mc_insertion_review.json`:

```json
{
  "decision": "approved_for_mc_insertion_bundle",
  "mc_review_pdf_sha256": "SHA256 of the actual mc_review.pdf",
  "mc_table_figure_pdf_sha256": "SHA256 of the actual mc_table_figure.pdf",
  "references_resolved": true,
  "checks": {
    "coverage": true,
    "statistics": true,
    "structural": true,
    "visual_table": true,
    "visual_figure": true,
    "qualifications": true
  },
  "acknowledges_mc_exceptions": true
}
```

`acknowledges_mc_exceptions` is required only when actual MC exceptions exist.
Finalization verifies both reviewed PDF hashes, build logs, prepared checksums,
and copied measurement/asset hashes. It does not rebuild the PDFs after review.

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/mc_insertion_bundle.py finalize
```

This creates only the separate `CVI_MC_Insertion_Bundle.zip` and adjacent
`.zip.sha256`; existing folders/archives are not silently overwritten. A qualified
bundle stays `reviewed_with_mc_exceptions`; review does not erase exceptions or
turn numerical objectives into formal gap certificates. The structural auditor
itself still performs zero independent gap solves.

`test_mc_insertion_bundle.py` uses explicitly labeled synthetic in-memory and
temporary LaTeX fixtures only. It checks the nonblocking single-gap exceptions,
MC-specific qualification, strict row coverage, audit gates, review hashes, and
standalone compilation. It reads no current measured case data and creates no
real bundle or delivered-paper change.
