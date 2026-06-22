"""
RETRACT — Publication-time topological early-warning for retraction.

Claim: a paper's citation/authorship neighbourhood AS IT EXISTS AT PUBLICATION TIME
carries a structural signature of eventual retraction that a metadata/text baseline
does not capture. The finding is *which* birth-time features carry the signal.

This module ships a fully-offline `--demo` that validates the pipeline end-to-end on a
synthetic scholarly cohort with a *planted* topological signal (so we can verify the
pipeline recovers a signature when one exists, and reports a null when it does not), plus
a documented `--real` path that joins Retraction Watch labels to the OpenAlex graph.

Run:
    python -m src.main --demo                 # offline synthetic pipeline validation
    python -m src.main --demo --no-signal      # negative control: planted signal removed
    python -m src.main --real --help           # see real-data instructions
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path
import numpy as np

LOG = logging.getLogger("retract")

# Feature groups -------------------------------------------------------------
METADATA_FEATURES = ["team_size", "ref_count", "venue_tier", "author_prior_papers", "field_id"]
TOPOLOGY_FEATURES = ["self_citation_rate", "retraction_proximity", "inst_diversity", "ego_density"]
ALL_FEATURES = METADATA_FEATURES + TOPOLOGY_FEATURES


def make_synthetic_cohort(n=6000, n_years=12, seed=0, planted_signal=True):
    """Synthetic scholarly cohort with publication-time features.

    Topological features carry a planted retraction signal *beyond* metadata when
    planted_signal=True. Each paper has a publication year used for the out-of-time split.
    """
    rng = np.random.default_rng(seed)
    year = rng.integers(0, n_years, size=n)
    field = rng.integers(0, 6, size=n)

    # Metadata (weakly predictive on its own)
    team_size = rng.poisson(4, size=n) + 1
    ref_count = rng.poisson(30, size=n) + 5
    venue_tier = rng.integers(1, 5, size=n)
    author_prior = rng.poisson(8, size=n)

    # Topological (publication-time ego-network) features in [0,1]-ish
    self_cite = np.clip(rng.beta(1.5, 8, size=n), 0, 1)            # self-citation rate
    retr_prox = np.clip(rng.beta(1.2, 12, size=n), 0, 1)          # cites soon-retracted work
    inst_div = np.clip(rng.beta(4, 2, size=n), 0, 1)              # institutional diversity
    ego_density = np.clip(rng.beta(2.5, 3, size=n), 0, 1)        # ego-graph density

    # Latent retraction propensity
    z = -3.2
    # small metadata effect (detection bias proxy: high venue slightly less retracted)
    z = z + 0.15 * (team_size - 4) / 4 - 0.20 * (venue_tier - 2.5)
    if planted_signal:
        # the discovery the pipeline should recover: birth-time topology drives retraction
        z = z + 2.4 * retr_prox + 1.6 * self_cite - 1.1 * inst_div + 0.8 * ego_density
    p = 1.0 / (1.0 + np.exp(-z))
    label = rng.binomial(1, p)

    X = np.column_stack([
        team_size, ref_count, venue_tier, author_prior, field,
        self_cite, retr_prox, inst_div, ego_density,
    ]).astype(float)
    return X, label.astype(int), year


def _auc_ap(y_true, scores):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def fit_eval(X, y, year, cols, split_year, seed=0):
    """Train GradientBoosting on the given feature columns; out-of-time evaluation."""
    from sklearn.ensemble import GradientBoostingClassifier
    idx = [ALL_FEATURES.index(c) for c in cols]
    tr = year < split_year
    te = year >= split_year
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8)
    clf.fit(X[tr][:, idx], y[tr])
    s = clf.predict_proba(X[te][:, idx])[:, 1]
    auc, ap = _auc_ap(y[te], s)
    imp = dict(zip(cols, [float(v) for v in clf.feature_importances_]))
    return {"auc": auc, "ap": ap, "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_pos_test": int(y[te].sum()), "importances": imp}


def run_demo(args):
    out = Path(args.out); (out).mkdir(parents=True, exist_ok=True)
    X, y, year = make_synthetic_cohort(n=args.n, seed=args.seed, planted_signal=not args.no_signal)
    split = args.split_year

    base = fit_eval(X, y, year, METADATA_FEATURES, split, seed=args.seed)
    full = fit_eval(X, y, year, ALL_FEATURES, split, seed=args.seed)
    delta = full["auc"] - base["auc"]

    # the "signature": topology importances in the full model
    topo_imp = {k: v for k, v in full["importances"].items() if k in TOPOLOGY_FEATURES}
    signature = sorted(topo_imp.items(), key=lambda kv: -kv[1])

    metrics = {
        "mode": "demo",
        "planted_signal": not args.no_signal,
        "prevalence": float(y.mean()),
        "split_year": split,
        "baseline_metadata": base,
        "full_metadata_plus_topology": full,
        "topology_auc_gain_out_of_time": delta,
        "signature_ranked": signature,
        "verdict": ("topology adds signal" if delta > 0.02 else
                    "no topological signal beyond metadata (null)"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    LOG.info("baseline AUC=%.3f  full AUC=%.3f  gain=%+.3f  -> %s",
             base["auc"], full["auc"], delta, metrics["verdict"])
    LOG.info("birth-time signature (topology importances): %s",
             ", ".join(f"{k}={v:.2f}" for k, v in signature))
    _plots(out, base, full, topo_imp)
    return metrics


def _plots(out, base, full, topo_imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        LOG.warning("matplotlib unavailable, skipping figures (%s)", e)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["metadata\nbaseline", "metadata+\ntopology"],
              [base["auc"], full["auc"]], color=["#9aa0a6", "#1a73e8"])
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("out-of-time AUC")
    ax[0].set_title("Does birth-time topology beat metadata?")
    for i, v in enumerate([base["auc"], full["auc"]]):
        ax[0].text(i, v + 0.005, f"{v:.3f}", ha="center")
    keys = list(topo_imp.keys()); vals = [topo_imp[k] for k in keys]
    order = np.argsort(vals)
    ax[1].barh([keys[i] for i in order], [vals[i] for i in order], color="#1a73e8")
    ax[1].set_title("Birth-time structural signature")
    ax[1].set_xlabel("feature importance")
    fig.tight_layout(); fig.savefig(out / "retract_signature.png", dpi=130)
    LOG.info("wrote %s", out / "retract_signature.png")


REAL_INSTRUCTIONS = """
REAL-DATA MODE (not yet automated — this is the documented pipeline):

1. Labels: download the Retraction Watch database CSV (free via the Crossref partnership).
   Keep DOI, RetractionDate, OriginalPaperDate, Reason, Subject. Drop reinstatements and
   non-work reasons (publisher error / duplicate) or analyse them as a separate stratum.

2. Graph: resolve each DOI to an OpenAlex work id; pull works + referenced_works +
   authorships + institutions + publication_date from the OpenAlex API or snapshot.

3. Controls: for each retracted paper sample non-retracted controls MATCHED on field,
   publication year, venue tier, and cited_by_count (neutralises the scrutiny confound).

4. Features (PUBLICATION-TIME ONLY — unit-test for leakage):
   - reference-side: ref age dist, venue diversity, self-citation rate,
     retraction_proximity (fraction of refs already/eventually retracted or author-linked),
   - author-side: collaboration-graph centrality at t=pub, prior co-authorship density,
     institutional diversity, author prior-retraction history.

5. Models: reproduce the metadata/text XGBoost baseline (Research Integrity & Peer Review
   2025, ~0.87) then a GraphSAGE/GAT (PyTorch Geometric) on sampled ego-subgraphs.
   Evaluate with an OUT-OF-TIME split (train retractions <= year T, test > T).

6. The finding = SHAP/ablation over structural features (the signature), framed as a
   positive-unlabeled LOWER BOUND. If topology does not beat the baseline out-of-time,
   publish the honest null.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="RETRACT publication-time retraction early-warning")
    ap.add_argument("--demo", action="store_true", help="run offline synthetic pipeline validation")
    ap.add_argument("--real", action="store_true", help="print real-data pipeline instructions")
    ap.add_argument("--no-signal", action="store_true", help="negative control: remove planted signal")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-year", type=int, default=8, help="out-of-time cutoff (train < split)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = ap.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(args.out) / "run.log", mode="w")])
    if args.real:
        print(REAL_INSTRUCTIONS); return 0
    if not args.demo:
        args.demo = True  # default
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
