"""Hermetic tests for rigor.utils — math_utils, encoding, align.

Each test group covers one module and is fully self-contained (no fixtures,
no external data).  Tests are intentionally simple so failures are obvious.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from rigor.utils import (
    NumpyEncoder,
    align_frame,
    align_series,
    clip_safe,
    coerce_float,
    format_number,
    format_pct,
    max_consecutive,
    safe_divide,
)

# ---------------------------------------------------------------------------
# 1. math_utils — safe_divide
# ---------------------------------------------------------------------------


class TestSafeDivide:
    def test_divide_by_zero_returns_default(self) -> None:
        assert safe_divide(1, 0) == 0.0

    def test_divide_by_zero_custom_default(self) -> None:
        assert safe_divide(1, 0, default=9) == 9

    def test_numpy_scalars(self) -> None:
        result = safe_divide(np.float64(2), np.float64(4))
        assert math.isclose(result, 0.5)

    def test_non_finite_result_returns_default(self) -> None:
        # inf / 1 = inf → non-finite → default
        assert safe_divide(float("inf"), 1.0, default=-1.0) == -1.0

    def test_normal_division(self) -> None:
        assert math.isclose(safe_divide(10.0, 4.0), 2.5)

    def test_nan_numerator_returns_default(self) -> None:
        assert safe_divide(float("nan"), 2.0, default=99.0) == 99.0


# ---------------------------------------------------------------------------
# 2. math_utils — max_consecutive and clip_safe
# ---------------------------------------------------------------------------


class TestMaxConsecutive:
    def test_known_streak(self) -> None:
        arr = np.array([False, True, True, True, False, True])
        assert max_consecutive(arr) == 3

    def test_all_false(self) -> None:
        assert max_consecutive(np.array([False, False, False])) == 0

    def test_all_true(self) -> None:
        assert max_consecutive(np.array([True, True, True, True])) == 4

    def test_empty(self) -> None:
        assert max_consecutive(np.array([], dtype=bool)) == 0

    def test_pandas_series(self) -> None:
        s = pd.Series([True, True, False, True, True, True])
        assert max_consecutive(s) == 3

    def test_single_true(self) -> None:
        assert max_consecutive(np.array([True])) == 1


class TestClipSafe:
    def test_nan_maps_to_lo(self) -> None:
        assert clip_safe(np.nan, 0, 1) == 0.0

    def test_inf_maps_to_lo(self) -> None:
        assert clip_safe(float("-inf"), 0.0, 1.0) == 0.0

    def test_value_above_hi(self) -> None:
        assert clip_safe(5.0, 0.0, 1.0) == 1.0

    def test_value_below_lo(self) -> None:
        assert clip_safe(-1.0, 0.0, 1.0) == 0.0

    def test_value_in_range(self) -> None:
        assert math.isclose(clip_safe(0.5, 0.0, 1.0), 0.5)


# ---------------------------------------------------------------------------
# 3. encoding — NumpyEncoder and coerce_float
# ---------------------------------------------------------------------------


class TestNumpyEncoder:
    def test_mixed_types_do_not_crash(self) -> None:
        obj = {
            "a": np.int64(3),
            "b": np.float64(1.5),
            "c": np.array([1, 2]),
            "d": float("nan"),
        }
        result = json.dumps(obj, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["a"] == 3
        assert math.isclose(parsed["b"], 1.5)
        assert parsed["c"] == [1, 2]
        assert parsed["d"] is None  # nan → null

    def test_numpy_int(self) -> None:
        assert json.loads(json.dumps(np.int64(42), cls=NumpyEncoder)) == 42

    def test_numpy_bool(self) -> None:
        assert json.loads(json.dumps(np.bool_(True), cls=NumpyEncoder)) is True

    def test_numpy_float_inf_to_null(self) -> None:
        assert json.loads(json.dumps(np.float64(float("inf")), cls=NumpyEncoder)) is None

    def test_pandas_timestamp(self) -> None:
        ts = pd.Timestamp("2024-01-15")
        result = json.loads(json.dumps(ts, cls=NumpyEncoder))
        assert "2024-01-15" in result


class TestCoerceFloat:
    def test_string_float(self) -> None:
        assert math.isclose(coerce_float("1.5"), 1.5)  # type: ignore[arg-type]

    def test_invalid_string_returns_default(self) -> None:
        assert coerce_float("x", default=None) is None

    def test_non_finite_returns_default(self) -> None:
        assert coerce_float(float("inf"), default=None) is None

    def test_numpy_scalar(self) -> None:
        assert math.isclose(coerce_float(np.float64(3.14)), 3.14)  # type: ignore[arg-type]

    def test_none_returns_default(self) -> None:
        assert coerce_float(None, default=-1.0) == -1.0  # type: ignore[arg-type]


class TestFormatHelpers:
    def test_format_pct_normal(self) -> None:
        assert format_pct(0.1234) == "12.34%"

    def test_format_pct_none(self) -> None:
        assert format_pct(None) == "—"

    def test_format_pct_nan(self) -> None:
        assert format_pct(float("nan")) == "—"

    def test_format_number_normal(self) -> None:
        assert format_number(1.2345) == "1.23"

    def test_format_number_nan(self) -> None:
        assert format_number(float("nan")) == "—"

    def test_format_number_dp(self) -> None:
        assert format_number(1.23456, dp=4) == "1.2346"


# ---------------------------------------------------------------------------
# 4. align — align_series and align_frame
# ---------------------------------------------------------------------------


class TestAlignSeries:
    def _make_pair(self) -> tuple[pd.Series, pd.Series]:
        idx_a = pd.date_range("2020-01-01", periods=5)
        idx_b = pd.date_range("2020-01-03", periods=5)
        a = pd.Series([1.0, float("nan"), 3.0, 4.0, 5.0], index=idx_a)
        b = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx_b)
        return a, b

    def test_equal_length_returned(self) -> None:
        a, b = self._make_pair()
        a_al, b_al = align_series(a, b)
        assert len(a_al) == len(b_al)

    def test_no_nan_in_output(self) -> None:
        a, b = self._make_pair()
        a_al, b_al = align_series(a, b)
        assert not a_al.isna().any()
        assert not b_al.isna().any()

    def test_inner_join_only_intersection(self) -> None:
        # a covers 2020-01-01..05, b covers 2020-01-03..07 → overlap 03..05
        # but idx_a[1] (2020-01-02) is NaN in a, so that row drops too
        a, b = self._make_pair()
        a_al, b_al = align_series(a, b)
        # Intersection of idx_a and idx_b is 2020-01-03 to 2020-01-05 (3 days)
        # a[2020-01-03]=3.0 (not NaN), so all 3 rows survive
        assert len(a_al) == 3

    def test_numpy_arrays_equal_length(self) -> None:
        a, b = self._make_pair()
        a_al, b_al = align_series(a, b)
        assert len(a_al.to_numpy()) == len(b_al.to_numpy())

    def test_identical_indexes_no_nan(self) -> None:
        idx = pd.date_range("2021-01-01", periods=4)
        a = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        b = pd.Series([5.0, 6.0, 7.0, 8.0], index=idx)
        a_al, b_al = align_series(a, b)
        assert len(a_al) == 4
        assert list(a_al.values) == [1.0, 2.0, 3.0, 4.0]


class TestAlignFrame:
    def test_equal_length_returned(self) -> None:
        idx_r = pd.date_range("2020-01-01", periods=6)
        idx_p = pd.date_range("2020-01-03", periods=6)
        returns = pd.Series([0.1, 0.2, float("nan"), 0.4, 0.5, 0.6], index=idx_r)
        panel = pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
            index=idx_p,
        )
        ret_al, pan_al = align_frame(returns, panel)
        assert len(ret_al) == len(pan_al)

    def test_no_nan_in_output(self) -> None:
        idx = pd.date_range("2020-01-01", periods=4)
        returns = pd.Series([0.1, float("nan"), 0.3, 0.4], index=idx)
        panel = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        ret_al, pan_al = align_frame(returns, panel)
        assert not ret_al.isna().any()
        assert not pan_al.isna().any().any()

    def test_column_names_preserved(self) -> None:
        idx = pd.date_range("2020-01-01", periods=3)
        returns = pd.Series([0.1, 0.2, 0.3], index=idx)
        panel = pd.DataFrame({"alpha": [1.0, 2.0, 3.0], "beta": [4.0, 5.0, 6.0]}, index=idx)
        _, pan_al = align_frame(returns, panel)
        assert list(pan_al.columns) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Regression: ensure public API is importable from top-level rigor.utils
# ---------------------------------------------------------------------------


def test_public_api_importable() -> None:
    from rigor.utils import (  # noqa: F401  (import-only check)
        NumpyEncoder,
        align_frame,
        align_mask,
        align_series,
        clip_safe,
        coerce_float,
        format_number,
        format_pct,
        max_consecutive,
        safe_divide,
    )
