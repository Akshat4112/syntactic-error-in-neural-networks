# Official attractor cohorts

Use `syntaxbench.attractors.load_attractor_bins(project_root)` after fetching
the pinned archive. Each key 0–5 corresponds to its official
`numpred.test.N` file; each row retains its original label and preamble.
The loader does not infer attractors from text or assume the files partition
the primary test set. Do not append these cohorts to the primary test data.

`membership_report(bins)` reports row denominators, exact labeled-record
duplicates, cross-cohort intersections and the unique union. Per-cohort
accuracy must use its own row denominator; do not sum denominators to claim
a unique pooled population. The report contains no raw sentences.

Fixture validation covers all six bins and shared records. Upstream membership
semantics and observed overlap counts still require verification on the pinned
archive before this task is considered complete.
