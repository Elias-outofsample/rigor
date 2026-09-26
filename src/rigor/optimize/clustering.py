"""Parameter clustering — how many *distinct* good regions does the grid have?

Cluster the top configs in normalised parameter space (k-means, or a quantile
fallback when scikit-learn is absent) and report a representative per cluster.
Several tight clusters of high-Sharpe configs means the edge has multiple robust
neighbourhoods; one diffuse cluster means the winner is isolated. Ported from
the prior research library's ``cluster_parameters``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cluster_parameters"]


def _numeric_param_keys(rows: list[dict]) -> list[str]:
    return [k for k in rows[0]["params"]
            if all(isinstance(r["params"].get(k), (int, float)) for r in rows)]


def cluster_parameters(rows: list[dict], *, top_pct: float = 0.20,
                       n_clusters: int | None = None) -> dict:
    """Cluster the top ``top_pct`` configs (ranked, best first) by their params."""
    keys = _numeric_param_keys(rows)
    if not keys:
        return {"note": "no numeric params to cluster"}
    top_n = max(2, int(len(rows) * top_pct))
    top = rows[:top_n]
    if len(top) < 2:
        return {"note": "too few configs to cluster"}

    x = np.array([[float(r["params"][k]) for k in keys] for r in top], dtype="float64")
    lo, hi = x.min(axis=0), x.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    xn = (x - lo) / rng

    k = n_clusters or max(2, int(np.sqrt(len(top))))
    k = min(k, len(top))
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(xn)
        centers_n = km.cluster_centers_
    except ImportError:
        labels = pd.qcut(xn[:, 0], q=k, labels=False, duplicates="drop").astype(int)
        centers_n = np.array([xn[labels == c].mean(axis=0) if (labels == c).any()
                              else xn.mean(axis=0) for c in range(int(labels.max()) + 1)])

    centers = [{key: round(float(c[i] * rng[i] + lo[i]), 6) for i, key in enumerate(keys)}
               for c in centers_n]
    sharpes = np.array([r["sharpe"] for r in top], dtype="float64")
    stats = []
    for cid in sorted({int(x) for x in labels}):
        m = labels == cid
        stats.append({"cluster": cid, "size": int(m.sum()),
                      "mean_sharpe": round(float(np.nanmean(sharpes[m])), 4),
                      "max_sharpe": round(float(np.nanmax(sharpes[m])), 4)})
    return {"n_clusters": len(centers), "top_n": top_n,
            "cluster_centers": centers, "cluster_stats": stats}
