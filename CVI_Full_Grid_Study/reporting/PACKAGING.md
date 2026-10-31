# Portable single-constraint LaTeX package

The helper performs no action until explicitly invoked. Existing adoption,
reporting, auditing, and assembly scripts remain unchanged.

From the workspace, after the measured single-only report/audits and article
assembly have passed:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/package_single_latex.py prepare
```

Defaults are `CVI_Full_Grid_Study/manuscript_single_ready` as source and
`CVI_Single_Constraint_LaTeX/` at workspace root as destination. The manuscript
is flattened at that root, with main.tex, proof files and JoC_submission/
directly accessible. Only `main.tex` and `numerical_review.tex` wrappers receive
ResultsDir path normalization to measured_results; source/copy hashes and both
before/after wrapper hashes are recorded. All theory and submission sources
stay byte-identical. Preparation does not create a ZIP or claim final review.

The folder includes measured_results/, the single-only data/ plan and all 60
saved cases, frozen code/, selected v1/v2 recovery evidence/source/backups,
reporting source, audit notes, a brief build README and checksums. It excludes
aux/log/cache junk and every old multi-constraint measurement directory.
Raw record provenance remains byte-identical; package_manifest.json maps its
original source paths to local copies and records unchanged proof hashes.

Build both portable entries:

```sh
cd CVI_Single_Constraint_LaTeX
sh build.sh
pdflatex -interaction=nonstopmode -halt-on-error numerical_review.tex
pdflatex -interaction=nonstopmode -halt-on-error numerical_review.tex
```

After inspecting the actual portable PDF, root must write
`CVI_Full_Grid_Study/notes/single_delivery_review.json` with:

```json
{
  "decision": "approved_for_portable_release",
  "package_main_pdf_sha256": "SHA256 of portable main.pdf",
  "references_resolved": true,
  "citations_resolved": true,
  "visual_inspection_passed": true,
  "checks": {
    "references": true,
    "citations": true,
    "visual_tables": true,
    "visual_figure": true,
    "proofs_unchanged": true
  }
}
```

Then, from the workspace:

```sh
.venv/bin/python CVI_Full_Grid_Study/reporting/package_single_latex.py finalize
```

Finalization checks the approved portable PDF hash, build logs, unchanged source,
measurement, candidate and proof hashes; adds the review evidence; refreshes
CHECKSUMS.sha256; and writes `CVI_Single_Constraint_LaTeX.zip` plus its adjacent
`.zip.sha256`. Existing folders/archives are not silently overwritten. The ZIP
contains one CVI_Single_Constraint_LaTeX/ folder with main.tex at that folder's
root. No compiler or solver is invoked by the helper itself.

`test_portable_package.py` tests path-only wrapper normalization, final-review
gates and junk exclusions without copying any measured dataset or creating the
real portable destination/archive.
