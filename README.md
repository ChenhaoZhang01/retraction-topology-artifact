# Born Flawed? Publication-time citation topology and retraction: replication artifact

This repository reproduces the measurements in the paper "Born Flawed? Publication-Time Citation Topology
Improves Retraction Early-Warning."

> Anonymized for review. Author and institution details are withheld. A permanent archived version with a
> DOI will be deposited on acceptance.

## Layout

```
src/        measurement harness: metadata baseline (census.py), publication-time topology (topology.py),
            and an offline synthetic reproduction (main.py)
tests/      stdlib unittest tests for the leakage invariant and the AUC evaluator (no network)
outputs/    metrics and figures reported in the paper
```

## Reproducing

Python 3.13 is recommended.

```bash
pip install -r requirements.txt
python -m unittest discover tests          # offline tests (leakage guard + signal detection)
python -m src.main --demo                  # offline synthetic pipeline check
python -m src.topology --n 300             # live run against OpenAlex (set MAIL in src/census.py first)
```

`src/census.py` defines `MAIL`, the contact address sent to the OpenAlex API for its polite pool. Set it to
your own email before running a live census. Live runs query OpenAlex, so exact counts shift over time as the
graph is updated; the committed `outputs/` are the snapshot used in the paper.

### Key results

| Result | File |
|---|---|
| Leakage-safe out-of-time AUC gain +0.174 (metadata vs. metadata+topology) | `outputs/metrics_topology.json` |
| Metadata-only baseline | `outputs/metrics_real.json` |

## Notes on method

- The headline uses only publication-time, leakage-safe features. A reference-retraction-proximity feature
  that can encode post-publication information is computed but excluded from the headline; the test in
  `tests/test_topology.py` guards that it never enters the clean feature set.
- Labels are positive-unlabeled (most flawed papers are never retracted), so reported gains are conservative.

## License

Code released under the MIT License. OpenAlex and the Retraction Watch database are public; their data are
not redistributed here.
