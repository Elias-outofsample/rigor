"""Frozen data releases: package, share, and verify an as-of data snapshot.

The reproducibility guarantee rests on everyone using the *same bytes*. One
person builds the cache online, then ``freeze``s it into a single checksummed
archive (``data_release_<as_of>.tar.gz``). Anyone can ``restore`` that archive and
run with ``offline=True`` (``rigor run --offline``) — no live fetches, identical
inputs, identical outputs. ``verify`` recomputes the snapshot hash so any two
people can prove their snapshots match.

The snapshot hash is the sha256 over the sorted ``<sha256>  <relpath>`` lines of
every cached file (excluding the manifest itself) — order-independent and
platform-independent.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .config import DEFAULT_CACHE_DIR

MANIFEST_NAME = "_MANIFEST.json"


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(snap_dir: Path) -> tuple[dict[str, str], str]:
    """Return ({relpath: sha256}, snapshot_sha256) for a snapshot directory."""
    files: dict[str, str] = {}
    for p in sorted(snap_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            files[p.relative_to(snap_dir).as_posix()] = _file_sha(p)
    blob = "\n".join(f"{files[k]}  {k}" for k in sorted(files)).encode()
    return files, hashlib.sha256(blob).hexdigest()


def freeze(as_of: str, cache_dir: Path = DEFAULT_CACHE_DIR, out_path: Path | None = None) -> dict:
    """Package ``data_cache/<as_of>/`` into a checksummed ``data_release`` archive."""
    snap_dir = Path(cache_dir) / as_of
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"no cached snapshot at {snap_dir}")
    files, snap = _manifest(snap_dir)
    (snap_dir / MANIFEST_NAME).write_text(
        json.dumps({"as_of": as_of, "snapshot_sha256": snap, "n_files": len(files),
                    "files": files}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out = Path(out_path) if out_path else Path(cache_dir).parent / f"data_release_{as_of}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(snap_dir, arcname=as_of)
    return {"archive": out, "snapshot_sha256": snap, "n_files": len(files),
            "size_bytes": out.stat().st_size}


def restore(archive: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Extract a data-release archive into ``cache_dir`` and verify its integrity."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        top = {m.name.split("/")[0] for m in tar.getmembers() if m.name}
        tar.extractall(cache_dir, filter="data")
    as_of = sorted(top)[0] if top else ""
    return verify(as_of, cache_dir)


def verify(as_of: str, cache_dir: Path = DEFAULT_CACHE_DIR, expected: str | None = None) -> dict:
    """Recompute the snapshot hash and compare to the recorded/expected hash."""
    snap_dir = Path(cache_dir) / as_of
    _, snap = _manifest(snap_dir)
    mpath = snap_dir / MANIFEST_NAME
    recorded = None
    if mpath.exists():
        recorded = json.loads(mpath.read_text(encoding="utf-8"))["snapshot_sha256"]
    ok = (recorded is None or snap == recorded) and (expected is None or snap == expected)
    return {"as_of": as_of, "snapshot_sha256": snap, "recorded": recorded,
            "expected": expected, "ok": bool(ok)}
