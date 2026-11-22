# Agreement baselines

The majority baseline learns the most frequent canonical label from official
training records only (0 singular, 1 plural); ties resolve to singular.
Predictions are constant and do not inspect validation or test labels.

The nearest-noun heuristic consumes independently Penn Treebank POS-tagged
preambles, scanning right to left for NN/NNP (singular) or NNS/NNPS (plural).
It abstains when no supported noun is found or the closest noun has an
unsupported noun tag. Pronouns are not treated as nouns. The heuristic does
not use gold subject position, target verb, or test labels. Tagging errors,
proper names and collective nouns can make it unreliable. An independently
versioned tagger and its configuration must be recorded before real evaluation;
this module does not download a tagger or claim raw-text tagging support.

Example (after explicit dataset retrieval):

```python
from pathlib import Path
from syntaxbench.baselines import MajorityBaseline, nearest_noun, coverage
from syntaxbench.validation import parse_record

with Path("data/rnn_agr_simple/numpred.train").open(encoding="utf-8") as stream:
    model = MajorityBaseline.fit(parse_record(line) for line in stream)
prediction = model.predict("the dogs near the tree")
heuristic = nearest_noun([("dogs", "NNS"), ("near", "IN"), ("tree", "NN")])
report = coverage([heuristic, None])
```

Report answered, abstained, total, coverage and accuracy among answered
records separately; never silently remove abstentions from the denominator.
Apply the same approach per official attractor cohort. Fixture tests verify
behavior but do not establish held-out benchmark scores.
