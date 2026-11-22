# Official dataset partitions

After fetching the pinned archive, regenerate all derived files together:

```sh
syntaxbench fetch
syntaxbench validate
syntaxbench preprocess
```

`data/train_df.csv`, `data/validation_df.csv` and `data/test_df.csv` now correspond to `numpred.train`, `numpred.val` and `numpred.test`, respectively. Older `test_df.csv` files contained validation data; do not reuse them. Regenerate derived data before training and do not compare old random-split metrics directly with official-split results. CSVs preserve raw preambles and labels (0=VBZ, 1=VBP), without a generated index column.

LSTM vocabulary and Word2Vec fitting use only training text. Validation is supplied separately to the LSTM and BERT trainers. The BERT test prediction runs after training. This does not establish full model reproducibility or checkpoint compatibility; those are separate tasks.

`data/split-overlap.json` records row counts and pairwise intersections of unique exact preambles, ignoring label for overlap detection. Upstream duplicates and overlaps are not removed, so official membership is preserved. Overlap is a limitation of the benchmark, not proof of independent evaluation examples. No raw text is included in the overlap report. The upstream verification workflow regenerates this report from the checksum-verified archive.
