# SYN-001 — Repository and dependency audit

Baseline: [main at 4fea4ca](https://github.com/Akshat4112/syntactic-error-in-neural-networks/commit/4fea4ca04bab4d47a9b631387b9b878753557b7b). Scope: static source, tracked tree, dependency manifest, tests and CI logs. No full training or statistical replication was performed.

## Inventory

| Area | Baseline evidence |
| --- | --- |
| CLI | [code/main.py](../code/main.py): preprocess, LSTM/BERT train, LSTM/RNN/BERT evaluate |
| Data | [dataset.py](../code/dataset.py), raw train/val/test and test.0–5, derived train/test CSVs |
| Models | [train.py](../code/train.py): Keras LSTM and HF BERT; [evaluation.py](../code/evaluation.py): human-data inference; standalone transformer script |
| Historical assets | notebook in data/, two LSTM plots, archived BERT script, recorded trainer/W&B logs |
| Packaging | [pyproject.toml](../pyproject.toml) declares project and pytest paths; [requirements.txt](../requirements.txt) is a legacy pinned environment |
| Tests | test_dataset.py and test_train.py; text cleaning exercises production code, several label/metric tests duplicate logic |
| CI | [ci.yml](../.github/workflows/ci.yml): Python 3.11, dependency install, flake8, pytest |

## Verified static findings and remediation ownership

| Finding | Evidence | Tracker |
| --- | --- | --- |
| CI red | Latest main run [34728012629](https://github.com/Akshat4112/syntactic-error-in-neural-networks/actions/runs/34728012629) failed; historical job [103645545368](https://github.com/Akshat4112/syntactic-error-in-neural-networks/actions/runs/28532273140/job/103645545368) logged F401 torch/logging and E302 in archive/BERT.py, F841 predictions in train.py. Tests did not follow failed lint in that job. | SYN-002 |
| Dependency gaps | train.py imports torch and evaluate, neither explicitly pinned in requirements; tensorflow-intel is platform-specific and CI filters it, pathtools and seqeval. README claims Python >=3.10; compatibility beyond CI's 3.11 is unverified. | SYN-003 |
| Packaging | Flat code directory, no declared CLI entry point or runtime dependencies in pyproject. | SYN-004 |
| Tracked generated material | IDE metadata, AppleDouble files, trainer events and W&B logs tracked; review privacy before any redistribution. | SYN-005 |
| Download stub | dataset.download_data is pass, despite CLI --download. | SYN-006 |
| Split semantics | dataset.py reads numpred.val into test_df.csv, not numpred.test. Raw README describes 9/1/90 split and homogeneous intervention test subsets 0–5; actual val filename differs from README's valid. Validate counts and source before claiming exact split sizes. | SYN-007–009 |
| Internal leakage | prepare_training fits Word2Vec and tokenizer to entire train CSV before internal split and model.fit validation split. This contaminates internal validation, not proof of leakage into official test. | SYN-008, SYN-013 |
| LSTM inference | Evaluation passes raw preamble strings to numerical embedding model; tokenizer is local to training and not persisted, no OOV policy specified. | SYN-013, SYN-015 |
| RNN path incomplete | evaluate_rnn loads legacy model_LSTM_2_epochs.h5; CLI has no RNN training option, no checkpoint in tree. | SYN-012 |
| BERT coupling | CLI runs LSTM preparation for both training choices; BERT uses internal unseeded-at-call split, predicts mislabeled validation CSV, discards predictions; tokenizer/revision provenance not explicitly managed. | SYN-016 |
| Behavioral comparison | Three raw and derived paradigm CSVs are present; evaluation writes prediction columns but no item-level alignment/statistical analysis in production evaluation. LSTM returns arrays, BERT label strings. Do not expose participant data until rights/privacy review. | SYN-031–035 |
| Reproducibility | TensorFlow seed and sklearn split seed exist, but no complete cross-library seed/config/run manifest; no verified checkpoint in tracked tree. | SYN-021–025 |
| Test gaps | test_train.py defines its own compute_metrics, and test_dataset.py duplicates label mapping rather than calling preprocess_data. | SYN-026, SYN-046 |
| Licensing | README claims MIT; no root LICENSE in baseline tree. Dataset and human data rights not established in this audit. | SYN-010, SYN-031, SYN-049 |

## Claims and validation boundaries

README reports approximately 93% LSTM after one epoch and 99% LSTM/BERT after longer training, explicitly as validation accuracy. Committed plots and historical notebook are evidence of prior experimentation, not independently reproduced metrics. Do not describe these as held-out official test results or evidence of human-like grammar. No full experiment, fresh install, or local pytest execution was undertaken for this documentation-only audit.

Verification method: inspect main tree and source files, compare import statements with requirements, inspect workflow and historical lint logs, compare README claims against actual split and evaluation code. The audit does not assert that every historical failure persists unchanged beyond the inspected baseline. Next task SYN-002 should repair CI without masking failures; the audit alone does not change runtime behavior.

Research source: [Linzen, Dupoux & Goldberg (2016)](https://aclanthology.org/Q16-1037/); [upstream agreement repository](https://github.com/TalLinzen/rnn_agreement).
