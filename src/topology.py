"""
RETRACT topology chunk: does PUBLICATION-TIME structural topology beat the metadata baseline?

We extract reference-side ego-network features computable at (or near) publication time and test
whether adding them to the metadata GBM improves out-of-time retraction prediction. This answers
28's core question directly with engineered features (simplicity first); a GraphSAGE on full
ego-subgraphs is the heavier follow-up.

Topology features (per paper, from its OpenAlex references):
  - self_citation_rate : fraction of references sharing >=1 author id with the paper
  - ref_year_std       : dispersion of reference publication years
  - ref_year_gap       : paper_year - mean(reference year)   [reliance on old vs fresh work]
  - retraction_proximity : fraction of references that are retracted   *** see LEAKAGE NOTE ***

LEAKAGE NOTE: `retraction_proximity` uses OpenAlex `is_retracted` on references, which may reflect
retractions that happened AFTER our paper was published. It is therefore reported separately, and
the headline "topology gain" is computed WITHOUT it (the clean, leakage-safe feature set).

Run from project root:   python -m src.topology --n 300
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests

from src.census import sample_works, features, FEATS, MAIL, OA

LOG = logging.getLogger("topology")
OUT = Path(__file__).resolve().parent.parent / "outputs"
TOPO_CLEAN = ["self_citation_rate", "ref_year_std", "ref_year_gap"]
TOPO_FULL = TOPO_CLEAN + ["retraction_proximity"]


def short_id(url):
    return url.rsplit("/", 1)[-1] if url else url


def ref_meta(ref_ids):
    """Batch-fetch reference metadata (<=50 ids/call): id, is_retracted, year, author ids."""
    out = {}
    for i in range(0, len(ref_ids), 50):
        chunk = ref_ids[i:i + 50]
        try:
            r = requests.get(OA, params={
                "filter": "ids.openalex:" + "|".join(chunk),
                "select": "id,is_retracted,publication_year,authorships",
                "per-page": 50, "mailto": MAIL}, timeout=30)
            for w in r.json().get("results", []):
                auth = {a.get("author", {}).get("id") for a in (w.get("authorships") or [])}
                out[short_id(w["id"])] = {
                    "retracted": bool(w.get("is_retracted")),
                    "year": w.get("publication_year"),
                    "authors": {a for a in auth if a}}
        except Exception:
            pass
    return out


def topo_features(w):
    refs = [short_id(x) for x in (w.get("referenced_works") or [])][:80]
    paper_authors = {a.get("author", {}).get("id") for a in (w.get("authorships") or [])}
    paper_authors = {a for a in paper_authors if a}
    pyear = w.get("publication_year") or 0
    if not refs:
        return {k: 0.0 for k in TOPO_FULL}, 0
    meta = ref_meta(refs)
    got = [meta[r] for r in refs if r in meta]
    n = len(got) or 1
    self_cites = sum(1 for m in got if m["authors"] & paper_authors)
    retr = sum(1 for m in got if m["retracted"])
    years = [m["year"] for m in got if m["year"]]
    ref_std = float(np.std(years)) if len(years) > 1 else 0.0
    ref_gap = float(pyear - np.mean(years)) if years and pyear else 0.0
    return {
        "self_citation_rate": self_cites / n,
        "retraction_proximity": retr / n,
        "ref_year_std": ref_std,
        "ref_year_gap": ref_gap,
    }, len(got)


def eval_models(X, y, years, cols_base, cols_full, label):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    idx_b = [ALLCOLS.index(c) for c in cols_base]
    idx_f = [ALLCOLS.index(c) for c in cols_full]
    clf = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    auc_b = roc_auc_score(y, cross_val_predict(clf, X[:, idx_b], y, cv=5, method="predict_proba")[:, 1])
    auc_f = roc_auc_score(y, cross_val_predict(clf, X[:, idx_f], y, cv=5, method="predict_proba")[:, 1])
    # out-of-time
    T = int(np.median(years)); tr, te = years < T, years >= T
    oot_b = oot_f = None
    if tr.sum() > 20 and te.sum() > 20 and len(set(y[te])) > 1:
        clf.fit(X[tr][:, idx_b], y[tr]); oot_b = roc_auc_score(y[te], clf.predict_proba(X[te][:, idx_b])[:, 1])
        clf.fit(X[tr][:, idx_f], y[tr]); oot_f = roc_auc_score(y[te], clf.predict_proba(X[te][:, idx_f])[:, 1])
    return {"cv_auc_base": float(auc_b), "cv_auc_full": float(auc_f), "cv_gain": float(auc_f - auc_b),
            "oot_auc_base": (float(oot_b) if oot_b else None),
            "oot_auc_full": (float(oot_f) if oot_f else None),
            "oot_gain": (float(oot_f - oot_b) if oot_b and oot_f else None)}


ALLCOLS = FEATS + TOPO_FULL


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT / "topology.log", mode="w")])

    LOG.info("sampling %d retracted + year-matched controls (with references)...", args.n)
    pos = sample_works("is_retracted:true", args.n, seed=0)
    pos = [w for w in pos if (w.get("publication_year") or 0) >= 1950 and w.get("referenced_works")]
    yrs = Counter(w["publication_year"] for w in pos)
    neg = []

    def controls(year, k):
        return [w for w in sample_works(f"is_retracted:false,publication_year:{year}", k * 2, seed=year)
                if w.get("referenced_works")][:k]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed({ex.submit(controls, y, k): y for y, k in yrs.items()}):
            try:
                neg += fut.result()
            except Exception:
                pass
    works = pos + neg
    LOG.info("computing topology features for %d papers (batch ref fetch)...", len(works))

    t0 = time.time(); topo = [None] * len(works)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(topo_features, w): i for i, w in enumerate(works)}
        for j, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                topo[i] = fut.result()[0]
            except Exception:
                topo[i] = {k: 0.0 for k in TOPO_FULL}
            if j % 100 == 0:
                LOG.info("  %d/%d (%.0fs)", j, len(works), time.time() - t0)

    rowsX, y, years = [], [], []
    for w, tf in zip(works, topo):
        mf = features(w)
        rowsX.append([mf[c] for c in FEATS] + [tf[c] for c in TOPO_FULL])
        y.append(mf["_retracted"]); years.append(mf["year"])
    X = np.array(rowsX, float); y = np.array(y); years = np.array(years)

    clean = eval_models(X, y, years, FEATS, FEATS + TOPO_CLEAN, "clean")
    full = eval_models(X, y, years, FEATS, FEATS + TOPO_FULL, "full")

    metrics = {
        "mode": "topology-vs-metadata", "n_retracted": int(y.sum()), "n_controls": int((1 - y).sum()),
        "headline_leakage_safe": {  # topology WITHOUT retraction_proximity
            "metadata_cv_auc": clean["cv_auc_base"], "metadata+topology_cv_auc": clean["cv_auc_full"],
            "cv_gain": clean["cv_gain"],
            "metadata_oot_auc": clean["oot_auc_base"], "metadata+topology_oot_auc": clean["oot_auc_full"],
            "oot_gain": clean["oot_gain"],
            "verdict": ("clean topology adds signal" if (clean["oot_gain"] or 0) > 0.01
                        else "no leakage-safe topology gain")},
        "with_retraction_proximity_LEAKY": {
            "cv_gain": full["cv_gain"], "oot_gain": full["oot_gain"],
            "note": "retraction_proximity may use post-publication retractions; do not headline"},
        "topology_features_clean": TOPO_CLEAN, "topology_features_full": TOPO_FULL,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(OUT / "metrics_topology.json", "w"), indent=2)
    h = metrics["headline_leakage_safe"]
    LOG.info("LEAKAGE-SAFE: metadata OOT AUC=%.3f  +topology=%.3f  gain=%+.3f -> %s",
             h["metadata_oot_auc"] or 0, h["metadata+topology_oot_auc"] or 0, h["oot_gain"] or 0, h["verdict"])
    LOG.info("(with leaky retraction_proximity) oot_gain=%+.3f [not headlined]", full["oot_gain"] or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
