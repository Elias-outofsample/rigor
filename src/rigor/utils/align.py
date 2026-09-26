"""Index-alignment helpers that prevent silent length-mismatch bugs.

The canonical failure mode this module eliminates: after an implicit inner join
two series appear aligned but have different lengths or unmatched NaN positions,
causing numpy operations (correlations, regressions) to silently compute on
misaligned data.  Every function here returns objects guaranteed to be
equal-length with no NaN values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_series(
    a: pd.Series,
    b: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Inner-join *a* and *b* on their indexes, then drop rows where either is NaN.

    The returned pair is guaranteed to have identical indexes and no NaN values,
    so ``.to_numpy()`` on each produces equal-length arrays safe for numpy ops
    (correlations, regressions, arithmetic).

    This prevents the silent length-mismatch class of bug that arises after an
    implicit inner join leaves rows where one series had NaN and the other did not.

    Parameters
    ----------
    a, b:
        Input series (any dtype).  Their indexes may differ in length, order,
        or type — only the intersection is kept.

    Returns
    -------
    (a_aligned, b_aligned):
        Pair of Series with the same, sorted index and no NaN rows.
    """
    combined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    combined = combined.dropna()
    return combined["a"], combined["b"]


def align_frame(
    returns: pd.Series,
    panel: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join *returns* and *panel* on their indexes, dropping any-NaN rows.

    Analogous to :func:`align_series` but for a Series vs a multi-column
    DataFrame.  Rows where *returns* OR any column of *panel* is NaN are
    dropped so that downstream matrix operations receive a clean, aligned block.

    Parameters
    ----------
    returns:
        1-D return series.
    panel:
        DataFrame whose columns represent features / factor exposures.

    Returns
    -------
    (returns_aligned, panel_aligned):
        Series and DataFrame sharing the same index with no NaN values.
    """
    combined = pd.concat(
        [returns.rename("__ret__"), panel],
        axis=1,
        join="inner",
    )
    combined = combined.dropna()
    ret_out = combined.pop("__ret__")
    return ret_out, combined


def align_mask(
    a: pd.Series,
    b: pd.Series,
    mask: pd.Series | np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Align *a* and *b*, then apply *mask* on the common index.

    Combines :func:`align_series` with a boolean-mask filter.  *mask* is
    broadcast to the inner-joined index before application, so it can be
    defined on a superset index without raising alignment errors.

    Parameters
    ----------
    a, b:
        Input series.
    mask:
        Boolean Series or array.  If a Series, its index is inner-joined with
        the already-aligned pair.  If a bare array it must be the same length
        as the aligned pair after NaN-dropping.

    Returns
    -------
    (a_masked, b_masked, mask_aligned):
        All three share the same index, have no NaN values, and only contain
        rows where *mask* is True.
    """
    a_al, b_al = align_series(a, b)

    if isinstance(mask, pd.Series):
        mask_al = mask.reindex(a_al.index).fillna(False).astype(bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if len(mask_arr) != len(a_al):
            msg = (
                f"mask length {len(mask_arr)} does not match aligned length {len(a_al)}; "
                "pass a pd.Series with a matching index instead"
            )
            raise ValueError(msg)
        mask_al = pd.Series(mask_arr, index=a_al.index)

    a_out = a_al[mask_al]
    b_out = b_al[mask_al]
    mask_out = mask_al[mask_al]
    return a_out, b_out, mask_out
