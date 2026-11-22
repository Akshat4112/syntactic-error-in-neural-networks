# Official attractor cohorts

Use `syntaxbench.attractors.load_attractor_bins(project_root)` after fetching the pinned archive. Each key 0–5 corresponds to its official `numpred.test.N` file; each row retains its original label and preamble. The loader does not infer attractors from text or assume the files partition the primary test set. Do not append these cohorts to the primary test data.

`membership_report(bins)` reports row denominators, exact labeled-record duplicates, cross-cohort intersections and the unique union. Per-cohort accuracy must use its own row denominator; do not sum denominators to claim a unique pooled population. The report contains no raw sentences.

## Verified archive audit

Measured on 2026-11-22 in [upstream verification CI](https://github.com/Akshat4112/syntactic-error-in-neural-networks/actions/runs/45046849385), head `3f4c97d05adddb28b7c51b3deac27c9d794caae2`.

| Attractors | Rows | Unique labeled records | Duplicate rows |
| --- | ---: | ---: | ---: |
| 0 | 1146331 | 754706 | 391625 |
| 1 | 52599 | 50433 | 2166 |
| 2 | 9380 | 9336 | 44 |
| 3 | 2051 | 2046 | 5 |
| 4 | 561 | 561 | 0 |
| 5 | 159 | 159 | 0 |

All unique labeled records in each cohort also occur in the primary test file (1,419,491 rows). The unique union across cohorts is 817,212 records. Bins 0 and 1 share 28 unique labeled records; bins 0 and 2 share one; all other pairwise intersections are zero. These are exact label/preamble comparisons, not original sentence identifiers or proof of syntactic equivalence. Preserve official row membership and report each cohort separately. Duplicate rows and cross-cohort overlap are inherited from the archive; these counts are not model performance metrics.

Fixture validation covers all six bins and shared records. The upstream workflow checks each denominator against the validation manifest.
