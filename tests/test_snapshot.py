import pytest

from rigor.data import snapshot
from rigor.data.cache import OfflineCacheMiss, RawCache


def _make_cache(tmp_path, as_of="2026-06-01"):
    c = RawCache(tmp_path, as_of)
    c.get_or_fetch("eod", "AAPL.US", lambda: [{"date": "2020-01-01", "close": 1.0}])
    c.get_or_fetch("splits", "AAPL.US", lambda: [])
    return tmp_path, as_of


def test_offline_serves_cache_but_raises_on_miss(tmp_path):
    online = RawCache(tmp_path, "2026-06-01")
    online.get_or_fetch("eod", "AAPL.US", lambda: [1])
    off = RawCache(tmp_path, "2026-06-01", offline=True)
    assert off.get_or_fetch("eod", "AAPL.US", lambda: [99]) == [1]   # from snapshot, no fetch
    with pytest.raises(OfflineCacheMiss):
        off.get_or_fetch("eod", "MSFT.US", lambda: [2])              # miss -> hard error


def test_freeze_verify_restore_roundtrip(tmp_path):
    cache_dir, as_of = _make_cache(tmp_path)
    out = tmp_path / "rel.tar.gz"
    info = snapshot.freeze(as_of, cache_dir=cache_dir, out_path=out)
    assert out.exists() and len(info["snapshot_sha256"]) == 64
    assert snapshot.verify(as_of, cache_dir=cache_dir, expected=info["snapshot_sha256"])["ok"]
    # restore into a fresh location -> identical hash
    res = snapshot.restore(out, cache_dir=tmp_path / "fresh")
    assert res["ok"] and res["snapshot_sha256"] == info["snapshot_sha256"]


def test_tamper_is_detected(tmp_path):
    cache_dir, as_of = _make_cache(tmp_path)
    snapshot.freeze(as_of, cache_dir=cache_dir, out_path=tmp_path / "rel.tar.gz")
    bad = next((cache_dir / as_of / "eod").glob("*.json"))
    bad.write_text("[]", encoding="utf-8")  # corrupt a cached file
    assert not snapshot.verify(as_of, cache_dir=cache_dir)["ok"]
