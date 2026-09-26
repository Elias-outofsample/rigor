"""Tests for cache schema versioning (sidecar .meta.json).

These tests are hermetic: they drive a RawCache pointed at tmp_path and never
touch the real data_cache directory.
"""

from __future__ import annotations

import json

import pytest

from rigor.data.cache import (
    CACHE_SCHEMA_VERSION,
    OfflineCacheMiss,
    RawCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache(tmp_path, *, offline: bool = False) -> RawCache:
    return RawCache(tmp_path, "2099-01-01", offline=offline)


def _counter(return_value):
    """Return a (fetch_fn, calls) pair; calls is a mutable list acting as counter."""
    calls: list[int] = []

    def fn():
        calls.append(1)
        return return_value

    return fn, calls


# ---------------------------------------------------------------------------
# Test 1: write creates blob + sidecar; second call is a cache hit
# ---------------------------------------------------------------------------

def test_first_fetch_writes_blob_and_meta(tmp_path):
    cache = _cache(tmp_path)
    fetch_fn, calls = _counter({"price": 42})

    result = cache.get_or_fetch("eod", "AAPL.US", fetch_fn)

    assert result == {"price": 42}
    assert len(calls) == 1, "fetch_fn should be called exactly once"

    blob = cache._path("eod", "AAPL.US")
    meta = cache._meta_path(blob)
    assert blob.exists(), "blob must be written"
    assert meta.exists(), "sidecar .meta.json must be written"

    meta_data = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_data["schema_version"] == CACHE_SCHEMA_VERSION
    assert meta_data["format"] == "eod"
    assert "created_at" in meta_data


def test_second_call_is_cache_hit(tmp_path):
    cache = _cache(tmp_path)
    fetch_fn, calls = _counter({"price": 42})

    cache.get_or_fetch("eod", "AAPL.US", fetch_fn)
    result2 = cache.get_or_fetch("eod", "AAPL.US", fetch_fn)

    assert result2 == {"price": 42}
    assert len(calls) == 1, "fetch_fn must NOT be called again on cache hit"


# ---------------------------------------------------------------------------
# Test 2: stale schema_version triggers a miss (online re-fetch and offline error)
# ---------------------------------------------------------------------------

def test_stale_schema_online_refetches(tmp_path):
    cache = _cache(tmp_path)
    fetch_fn, calls = _counter({"price": 1})
    cache.get_or_fetch("eod", "MSFT.US", fetch_fn)
    assert len(calls) == 1

    # Corrupt the sidecar with a different schema_version.
    blob = cache._path("eod", "MSFT.US")
    meta_path = cache._meta_path(blob)
    stale = {"schema_version": CACHE_SCHEMA_VERSION + 99, "created_at": "x", "format": "eod"}
    meta_path.write_text(json.dumps(stale), encoding="utf-8")

    fetch_fn2, calls2 = _counter({"price": 2})
    result = cache.get_or_fetch("eod", "MSFT.US", fetch_fn2)

    assert result == {"price": 2}, "should re-fetch and return new data"
    assert len(calls2) == 1, "fetch_fn must be called once on schema miss"


def test_stale_schema_offline_raises(tmp_path):
    # Populate the blob online first.
    online = _cache(tmp_path)
    fetch_fn, _ = _counter({"price": 1})
    online.get_or_fetch("eod", "GOOG.US", fetch_fn)

    # Corrupt the sidecar.
    blob = online._path("eod", "GOOG.US")
    meta_path = online._meta_path(blob)
    stale = {"schema_version": CACHE_SCHEMA_VERSION + 99, "created_at": "x", "format": "eod"}
    meta_path.write_text(json.dumps(stale), encoding="utf-8")

    offline = _cache(tmp_path, offline=True)
    with pytest.raises(OfflineCacheMiss):
        offline.get_or_fetch("eod", "GOOG.US", lambda: None)


# ---------------------------------------------------------------------------
# Test 3: legacy blob (no sidecar) is treated as a cache hit
# ---------------------------------------------------------------------------

def test_legacy_blob_no_sidecar_is_hit(tmp_path):
    """A blob present without a sidecar (pre-versioning cache) must be served
    as-is -- we never nuke the existing 4 GB on-disk cache on upgrade."""
    cache = _cache(tmp_path)

    # Write the blob manually, simulating a pre-versioning cache entry.
    blob = cache._path("eod", "IBM.US")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(json.dumps({"price": 7}), encoding="utf-8")
    # Explicitly confirm the sidecar does NOT exist.
    meta_path = cache._meta_path(blob)
    assert not meta_path.exists(), "test setup: sidecar must be absent"

    fetch_fn, calls = _counter({"price": 99})
    result = cache.get_or_fetch("eod", "IBM.US", fetch_fn)

    assert result == {"price": 7}, "legacy blob must be served from cache"
    assert len(calls) == 0, "fetch_fn must NOT be called for legacy blob"


# ---------------------------------------------------------------------------
# Test 4: sidecar write is atomic (no leftover .tmp after success)
# ---------------------------------------------------------------------------

def test_sidecar_write_is_atomic_no_leftover_tmp(tmp_path):
    cache = _cache(tmp_path)
    fetch_fn, _ = _counter({"price": 5})
    cache.get_or_fetch("eod", "TSLA.US", fetch_fn)

    blob = cache._path("eod", "TSLA.US")
    endpoint_dir = blob.parent
    tmp_files = list(endpoint_dir.glob(".meta-*.tmp"))
    assert tmp_files == [], f"leftover temp files found: {tmp_files}"
