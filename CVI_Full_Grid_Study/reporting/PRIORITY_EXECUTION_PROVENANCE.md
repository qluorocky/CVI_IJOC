# Temporary ten-worker MC priority phase

The original worker source and experiment plan remain unchanged. The temporary
n=100,m=100 priority phase raises the admission ceiling from seven to ten
single-threaded workers; the unchanged regular-grid plan retains its seven-worker
ceiling. This is a resource-workload difference from already measured PFAOS data,
not a claim of matched concurrency or ten simultaneous CPUs.

`priority_execution_context.py` reads completed per-case `priority_execution.json`
sidecars. It binds the exact sidecar, immutable phase plan, original plan,
priority runner snapshot and scheduled timed run/point hashes. It checks the
setting, ten-worker/seven-worker limits, numerical-thread count, recorded pool
semantics, observed counts and the disjoint/exhaustive method split. Previously
committed methods retain their recorded earlier context rather than being
relabeled as new ten-worker measurements. `prior_execution` is preserved.

The two-instance plotter adds these contexts to its hash-bound provenance and
caption. Its `make_figure`, style constants, axes, fonts, strokes, markers,
legend coordinates and statistical aggregation are unchanged. The regular
full-grid reporter also hashes the sidecars and emits a data-derived priority
execution paragraph. The untimed assembler inserts that paragraph and qualifies
the generic seven-worker sentence as the regular-grid policy when overrides
exist. No already generated report, delivered article or ZIP is edited.

Selected copied snapshots can use a distinct subset plan, provided the phase
plan identifies the unchanged original plan bytes (or the subset plan links the
source full-plan SHA). Production plotting still requires completed, audited
ten-seed/five-method settings. Sidecar metadata is read only after completion;
scheduled pairs must exist and match their saved hashes. Configured ceilings and
observed peak process counts remain separate fields.

`test_priority_execution_context.py` uses synthetic temporary byte fixtures only
to test hash binding, unchanged original plans, method splits, observed-count
rejection, subset links and full-protocol disclosure. It launches no worker and
changes no actual experiment or delivered artifact.
