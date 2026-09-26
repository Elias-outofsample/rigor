"""``rigor demo``: the validation battery must reject a strategy tuned on pure noise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigor.data.synthetic import SyntheticDataLoader
from rigor.demo import run_demo


def test_synthetic_loader_is_deterministic_and_driftless():
    a = SyntheticDataLoader().prices("NOISE")
    b = SyntheticDataLoader().prices("NOISE")
    pd.testing.assert_frame_equal(a, b)
    # Same shocks, different drift: the paths differ by exactly the drift per bar, so the
    # default path carries no built-in trend (any realised trend is sampling noise).
    trending = SyntheticDataLoader(drift_annual=0.25).prices("NOISE")
    gap = np.diff(np.log(trending["close"].to_numpy())) - np.diff(np.log(a["close"].to_numpy()))
    np.testing.assert_allclose(gap, 0.25 / 252, atol=1e-12)


def test_demo_rejects_noise_and_writes_artifacts(tmp_path, capsys):
    assert run_demo(tmp_path) == 0
    out = capsys.readouterr().out
    assert "Verdict on the winner: REJECT" in out
    for suffix in ("report.html", "summary.json", "thesis.pdf"):
        assert (tmp_path / f"ma_crossover_{suffix}").stat().st_size > 0
