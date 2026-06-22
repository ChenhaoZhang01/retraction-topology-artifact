"""
REAL first-chunk pipeline for RETRACT: publication-time METADATA baseline (the number the
topological GNN must beat). Positives = OpenAlex works with is_retracted:true; controls =
year-matched is_retracted:false works. Features use only publication-time-available signals
(no post-pub citations). Also downloads the Retraction Watch CSV (gold label + dates/reasons)
for the full run.

Full study (next chunk): join Retraction Watch DOIs, build publication-time ego-networks, train
a GraphSAGE on topology, evaluate out-of-time vs this baseline.

Usage:
    python -m src.census --n 500
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests

LOG = logging.getLogger("retract_census")
OUT = Path(__file__).resolve().parent.parent / "outputs"
MAIL = "anonymous@example.com"  # set to your email when running (OpenAlex polite pool)
OA = "https://api.openalex.org/works"
SELECT = "id,publication_year,referenced_works,authorships,title,is_retracted,cited_by_count,language"


def download_retraction_watch():
    try:
        r = requests.get(f"https://api.labs.crossref.org/data/retractionwatch?{MAIL}", timeout=120)
        if r.status_code == 200 and r.content[:9] == b"Record ID":
            (OUT / "retraction_watch.csv").write_bytes(r.content)
            n = r.content.count(b"\n")
            LOG.info("saved Retraction Watch CSV (%d rows) for the full run", n)
            return n
    except Exception as e:
        LOG.warning("RW download failed (non-fatal for the baseline): %s", e)
    return None


def sample_works(filt, n, seed=0):
    """Sample up to n works matching an OpenAlex filter (full objects, selected fields)."""
    works, page = [], 1
    while len(works) < n and page <= 25:
        try:
            r = requests.get(OA, params={"filter": filt, "sample": min(n, 200), "per-page": 200,
                                         "seed": seed, "select": SELECT, "mailto": MAIL}, timeout=40)
            if r.status_code != 200:
                break
            batch = r.json().get("results", [])
            if not batch:
                break
            works += batch
            if len(batch) < 200:
                break
            page += 1; seed += 1
        except Exception:
            break
    return works[:n]


def features(w):
    auth = w.get("authorships", []) or []
    insts, countries = set(), set()
    for a in auth:
        for inst in (a.get("institutions") or []):
            if inst.get("id"):
                insts.add(inst["id"])
        for c in (a.get("countries") or []):
            countries.add(c)
    return {
        "year": w.get("publication_year") or 0,
        "n_refs": len(w.get("referenced_works") or []),
        "n_authors": len(auth),
        "n_institutions": len(insts),
        "n_countries": len(countries),
        "title_len": len((w.get("title") or "").split()),
        "is_english": 1 if w.get("language") == "en" else 0,
        # cited_by_count is POST-publication -> kept out of features (leakage); matching only
        "_cited": w.get("cited_by_count") or 0,
        "_retracted": 1 if w.get("is_retracted") else 0,
    }


FEATS = ["year", "n_refs", "n_authors", "n_institutions", "n_countries", "title_len", "is_english"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT / "census.log", mode="w")])
    download_retraction_watch()

    LOG.info("sampling %d retracted works from OpenAlex...", args.n)
    pos = sample_works("is_retracted:true", args.n, seed=args.seed)
    pos_f = [features(w) for w in pos]
    pos_f = [f for f in pos_f if f["year"] and f["n_refs"] >= 0]
    yrs = Counter(f["year"] for f in pos_f if f["year"] >= 1950)
    LOG.info("got %d retracted with usable metadata; year range %s-%s",
             len(pos_f), min(yrs) if yrs else "?", max(yrs) if yrs else "?")

    # year-matched controls
    LOG.info("sampling year-matched non-retracted controls...")
    neg_f = []
    def get_controls(year, k):
        ws = sample_works(f"is_retracted:false,publication_year:{year}", k * 2, seed=year)
        return [features(w) for w in ws][:k]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(get_controls, y, k): y for y, k in yrs.items() if y >= 1950}
        for fut in as_completed(futs):
            try:
                neg_f += fut.result()
            except Exception:
                pass
    LOG.info("got %d controls", len(neg_f))

    # build matrix
    rows = pos_f + neg_f
    X = np.array([[f[k] for k in FEATS] for f in rows], float)
    y = np.array([f["_retracted"] for f in rows])
    years = np.array([f["year"] for f in rows])

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    clf = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3, learning_rate=0.05)
    cv_scores = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y, cv_scores))
    # out-of-time split
    T = int(np.median(years))
    tr, te = years < T, years >= T
    oot_auc = None
    if tr.sum() > 20 and te.sum() > 20 and len(set(y[te])) > 1:
        clf.fit(X[tr], y[tr])
        oot_auc = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    clf.fit(X, y)
    imp = dict(sorted(zip(FEATS, [float(v) for v in clf.feature_importances_]), key=lambda kv: -kv[1]))

    metrics = {
        "mode": "real-baseline-chunk1",
        "n_retracted": len(pos_f), "n_controls": len(neg_f),
        "metadata_baseline_cv_auc": cv_auc,
        "metadata_baseline_out_of_time_auc": oot_auc, "oot_split_year": T,
        "feature_importances": imp,
        "features_used": FEATS,
        "note": ("This is the publication-time METADATA baseline the topological GNN must beat. "
                 "Retraction Watch CSV saved as gold labels for the full ego-network + GraphSAGE run."),
    }
    json.dump(metrics, open(OUT / "metrics_real.json", "w"), indent=2)
    LOG.info("DONE: metadata baseline CV-AUC=%.3f | out-of-time AUC=%s (split %d)",
             cv_auc, round(oot_auc, 3) if oot_auc else None, T)
    LOG.info("top features: %s", list(imp.items())[:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
