# Dataset cards and usage restrictions

Review date: 2026-11-22. This document records provenance, not a grant of rights.

## Linzen agreement benchmark

Source: [upstream README at pinned revision](https://github.com/TalLinzen/rnn_agreement/blob/0001b91aab94c0f834f354a04ed22ba15cefd8da/README.md). The authors describe a simple dependency dataset and request academic citation. Cite Tal Linzen, Emmanuel Dupoux and Yoav Goldberg (2016), *Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies*, TACL 4, 521–535, [paper](https://aclanthology.org/Q16-1037/).

- Origin: parsed Wikipedia dependencies; English subject–verb number prediction from a preamble.
- Archive: https://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz
- SHA-256: `3e0e0c194696dc53378791ac9a75a35d134f4de0dc9253de7cc0a5c8e48b3142` (see `src/syntaxbench/fetch.py`; measured in [fetch verification](https://github.com/Akshat4112/syntactic-error-in-neural-networks/actions/runs/44922540789)).
- Official files: numpred.train, numpred.val, numpred.test and numpred.test.0 through numpred.test.5; labels VBZ = 0, VBP = 1. Fit preprocessing on train only, tune on validation, report held-out test separately. See [attractor subsets](attractor-subsets.md) for measured denominators and overlapping records; do not pool overlapping cohorts as unique observations.
- Run `syntaxbench fetch` then `syntaxbench validate` to obtain a per-file checksum, counts and duplicate manifest. Preserve raw files separately from derived outputs; do not commit newly downloaded records.

### Terms and limitations

The pinned upstream README provides a download and citation request but no explicit redistribution license. A request for the upstream root LICENSE at the pinned revision returned 404; this is not proof that no terms exist elsewhere. Wikipedia source attribution and applicable terms may also matter. Redistribution of raw/derived datasets and publication of checkpoints remain **unresolved** pending documented rights review. Do not infer permission from download availability, citation, or the software license. Existing tracked copies are not evidence of authorization; audit them before release and arrange any necessary remediation with the owner.

This benchmark measures a specific binary agreement task, not general grammatical competence. Record overlap and imbalance, and distinguish official primary test results from cohort results and smoke runs.

## Human behavioral data (restricted; not cleared)

The existing repository refers to SPEEDED_RSVP, SPEEDED_SPR and UNSPEEDED paradigms under `data/human_behavior_data/`. Original study citation, collection conditions, consent, license, participant/item schema, units and redistribution rights have not been established in this review. Do not infer these from filenames. Do not redistribute, train or evaluate against these records until the owner supplies authoritative study documentation and permission for the intended use (SYN-031). Do not expose raw participant identifiers, rows or reaction times in logs, demos or reports.

After clearance, distinguish grammatical target, human correctness and model prediction; verify participant/item keys, repeated trials, units and exclusions. Publish only permitted aggregate statistics, with uncertainty and disclosure review. Keep private input outside public artifacts.

## Software and release gate

The project README asserts MIT, but a root LICENSE request on main returned 404 during this review. Confirm authorization and the actual license text in SYN-049 rather than treating the claim as verified. This documentation does not relicense existing material. SYN-010 documents unresolved restrictions; SYN-031, SYN-045 and SYN-049 must enforce relevant permission gates before human-data analysis, checkpoint publication or release.
