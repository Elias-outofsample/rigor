"""Hermetic unit tests for forecast-error metrics in signal_quality.py.

Covers forecast_mae, forecast_rmse, forecast_mape, forecast_error_report.
Uses only numpy / pandas / pytest — no heavy optional dependencies.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rigor.analysis.signal_quality import (
    forecast_error_report,
    forecast_mae,
    forecast_mape,
    forecast_rmse,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PRED_BASIC = [1.0, 2.0, 3.0]
ACTUAL_BASIC = [1.5, 1.5, 4.0]
# Errors: |0.5|, |0.5|, |1.0|
# MAE  = (0.5 + 0.5 + 1.0) / 3 = 2.0/3
# RMSE = sqrt((0.25 + 0.25 + 1.0) / 3) = sqrt(1.5/3) = sqrt(0.5)
# MAPE = mean(|0.5/1.5|, |0.5/1.5|, |1.0/4.0|)
#       = mean(1/3, 1/3, 0.25) = (2/3 + 0.25) / 3 = (0.6667 + 0.25)/3 ≈ 0.30556
_MAE_EXPECTED = 2.0 / 3.0
_RMSE_EXPECTED = math.sqrt(0.5)
_MAPE_EXPECTED = (1.0 / 3.0 + 1.0 / 3.0 + 0.25) / 3.0


# ---------------------------------------------------------------------------
# forecast_mae
# ---------------------------------------------------------------------------

class TestForecastMAE:
    def test_basic_lists(self):
        result = forecast_mae(PRED_BASIC, ACTUAL_BASIC)
        assert result == pytest.approx(_MAE_EXPECTED, rel=1e-9)

    def test_basic_numpy(self):
        result = forecast_mae(np.array(PRED_BASIC), np.array(ACTUAL_BASIC))
        assert result == pytest.approx(_MAE_EXPECTED, rel=1e-9)

    def test_perfect_forecast(self):
        assert forecast_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_nan_pairs_dropped(self):
        # Third pair is NaN in predictions — should be dropped; result == mean(|0.5|, |0.5|) = 0.5
        p = [1.0, 2.0, float("nan")]
        a = [1.5, 1.5, 4.0]
        assert forecast_mae(p, a) == pytest.approx(0.5)

    def test_nan_in_actuals_dropped(self):
        p = [1.0, 2.0, 3.0]
        a = [1.5, float("nan"), 4.0]
        # Only pairs (1,1.5) and (3,4) valid → errors 0.5, 1.0 → MAE 0.75
        assert forecast_mae(p, a) == pytest.approx(0.75)

    def test_all_nan_returns_nan(self):
        result = forecast_mae([float("nan")], [float("nan")])
        assert math.isnan(result)

    def test_series_index_alignment(self):
        # Predictions on index [0,1,2], actuals on [1,2,3] → inner join [1,2]
        p = pd.Series([10.0, 1.0, 2.0], index=[0, 1, 2])
        a = pd.Series([1.5, 1.5, 99.0], index=[1, 2, 3])
        # Aligned: pred=[1,2], actual=[1.5,1.5] → errors 0.5, 0.5 → MAE 0.5
        assert forecast_mae(p, a) == pytest.approx(0.5)

    def test_length_mismatch_arrays_raises(self):
        with pytest.raises(ValueError, match="same length"):
            forecast_mae([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_negative_returns(self):
        # Errors are absolute: |(-1) - (-2)| = 1, |(-2) - (-1)| = 1 → MAE 1.0
        assert forecast_mae([-1.0, -2.0], [-2.0, -1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# forecast_rmse
# ---------------------------------------------------------------------------

class TestForecastRMSE:
    def test_basic(self):
        result = forecast_rmse(PRED_BASIC, ACTUAL_BASIC)
        assert result == pytest.approx(_RMSE_EXPECTED, rel=1e-9)

    def test_rmse_ge_mae(self):
        # RMSE >= MAE always
        mae = forecast_mae(PRED_BASIC, ACTUAL_BASIC)
        rmse = forecast_rmse(PRED_BASIC, ACTUAL_BASIC)
        assert rmse >= mae - 1e-12

    def test_perfect_forecast(self):
        assert forecast_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_nan_pairs_dropped(self):
        p = [1.0, 2.0, float("nan")]
        a = [1.5, 1.5, 4.0]
        # (0.5^2 + 0.5^2) / 2 = 0.25 → sqrt = 0.5
        assert forecast_rmse(p, a) == pytest.approx(0.5)

    def test_all_nan_returns_nan(self):
        assert math.isnan(forecast_rmse([float("nan")], [float("nan")]))

    def test_rmse_penalises_outlier_more_than_mae(self):
        # Uniform error 1: MAE=RMSE=1.  Replace last with large error 10.
        p1 = [0.0, 0.0, 0.0]
        a1 = [1.0, 1.0, 1.0]  # uniform
        p2 = [0.0, 0.0, 0.0]
        a2 = [1.0, 1.0, 10.0]  # outlier
        # RMSE should grow more than MAE relative to baseline
        mae_diff = forecast_mae(p2, a2) - forecast_mae(p1, a1)
        rmse_diff = forecast_rmse(p2, a2) - forecast_rmse(p1, a1)
        assert rmse_diff > mae_diff

    def test_series_alignment(self):
        # pred index [0,1,2] value [1.0, 2.0, 99.0]
        # actual index [1,2,3] value [2.5, 99.0, 50.0]
        # inner join on [1,2]: pred=[2.0,99.0], actual=[2.5,99.0]
        # errors: 0.5, 0.0 → RMSE = sqrt((0.25+0)/2) = sqrt(0.125)
        p = pd.Series([1.0, 2.0, 99.0], index=[0, 1, 2])
        a = pd.Series([2.5, 99.0, 50.0], index=[1, 2, 3])
        assert forecast_rmse(p, a) == pytest.approx(math.sqrt(0.125), rel=1e-9)


# ---------------------------------------------------------------------------
# forecast_mape
# ---------------------------------------------------------------------------

class TestForecastMAPE:
    def test_basic(self):
        result = forecast_mape(PRED_BASIC, ACTUAL_BASIC)
        assert result == pytest.approx(_MAPE_EXPECTED, rel=1e-9)

    def test_perfect_forecast(self):
        assert forecast_mape([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_near_zero_actuals_skipped(self):
        # actual=0 should be excluded; only the non-zero pair contributes
        p = [1.0, 1.0]
        a = [0.0, 2.0]  # first actual is zero → excluded
        # Only (1.0, 2.0) used: |(2-1)/2| = 0.5
        assert forecast_mape(p, a) == pytest.approx(0.5)

    def test_all_near_zero_actuals_returns_nan(self):
        result = forecast_mape([1.0, 2.0], [0.0, 0.0])
        assert math.isnan(result)

    def test_custom_zero_threshold(self):
        # With threshold=0.5, actual=0.3 is treated as near-zero and skipped
        p = [1.0, 1.0]
        a = [0.3, 2.0]
        result = forecast_mape(p, a, zero_threshold=0.5)
        # Only (1.0, 2.0) used: |(2-1)/2| = 0.5
        assert result == pytest.approx(0.5)

    def test_nan_pairs_dropped(self):
        p = [1.0, float("nan"), 3.0]
        a = [1.5, 1.5, 4.0]
        # Middle pair dropped; (1,1.5): |0.5/1.5|=1/3; (3,4): |1/4|=0.25 → mean=7/24
        expected = (1.0 / 3.0 + 0.25) / 2.0
        assert forecast_mape(p, a) == pytest.approx(expected, rel=1e-9)

    def test_all_nan_returns_nan(self):
        assert math.isnan(forecast_mape([float("nan")], [float("nan")]))

    def test_non_negative(self):
        result = forecast_mape([-0.01, -0.02], [-0.02, -0.01])
        assert result >= 0.0


# ---------------------------------------------------------------------------
# forecast_error_report
# ---------------------------------------------------------------------------

class TestForecastErrorReport:
    def test_basic_dict_keys(self):
        report = forecast_error_report(PRED_BASIC, ACTUAL_BASIC)
        assert set(report.keys()) == {"mae", "rmse", "mape", "n", "n_mape"}

    def test_basic_values(self):
        report = forecast_error_report(PRED_BASIC, ACTUAL_BASIC)
        assert report["mae"] == pytest.approx(_MAE_EXPECTED, rel=1e-9)
        assert report["rmse"] == pytest.approx(_RMSE_EXPECTED, rel=1e-9)
        assert report["mape"] == pytest.approx(_MAPE_EXPECTED, rel=1e-9)
        assert report["n"] == 3
        assert report["n_mape"] == 3

    def test_n_mape_less_than_n_when_zero_actuals(self):
        p = [1.0, 1.0, 1.0]
        a = [0.0, 1.5, 4.0]  # first actual near-zero → excluded from mape
        report = forecast_error_report(p, a)
        assert report["n"] == 3
        assert report["n_mape"] == 2

    def test_empty_inputs_returns_nan_report(self):
        report = forecast_error_report(
            [float("nan")], [float("nan")]
        )
        assert math.isnan(report["mae"])
        assert math.isnan(report["rmse"])
        assert math.isnan(report["mape"])
        assert report["n"] == 0
        assert report["n_mape"] == 0

    def test_consistent_with_individual_functions(self):
        report = forecast_error_report(PRED_BASIC, ACTUAL_BASIC)
        assert report["mae"] == pytest.approx(
            forecast_mae(PRED_BASIC, ACTUAL_BASIC), rel=1e-12
        )
        assert report["rmse"] == pytest.approx(
            forecast_rmse(PRED_BASIC, ACTUAL_BASIC), rel=1e-12
        )
        assert report["mape"] == pytest.approx(
            forecast_mape(PRED_BASIC, ACTUAL_BASIC), rel=1e-12
        )

    def test_series_input(self):
        p = pd.Series(PRED_BASIC)
        a = pd.Series(ACTUAL_BASIC)
        report = forecast_error_report(p, a)
        assert report["mae"] == pytest.approx(_MAE_EXPECTED, rel=1e-9)
        assert report["n"] == 3
