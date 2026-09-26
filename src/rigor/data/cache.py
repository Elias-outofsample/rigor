"""Local, reproducible cache of raw EODHD responses.

Layout:  ``<cache_dir>/<as_of>/<endpoint>/<symbol>.json``

Keying by as-of date is deliberate: a pull pinned to "2026-06-01" always lands
in the same place and is reused, so machines sharing an as-of date (or a seeded
snapshot of this folder) get byte-identical inputs. The cache stores the vendor
payload verbatim -- no transformation -- so it is a faithful, frozen record.

Schema versioning
-----------------
Every blob is accompanied by a sidecar ``<blobname>.meta.json`` written
atomically (temp file + os.replace) at the same time as the blob itself.
The sidecar contains::

    {"schema_version": <int>, "created_at": "<ISO-8601>", "format": "<kind>"}

On read, the sidecar is checked:

* **Sidecar absent** (legacy cache written before this feature) → treated as
  current version.  We never nuke the 4 GB on-disk cache for a tooling upgrade;
  only an explicit mismatch invalidates.
* **Sidecar present, schema_version == CACHE_SCHEMA_VERSION** → cache hit.
* **Sidecar present, schema_version != CACHE_SCHEMA_VERSION** → cache miss;
  re-fetch live (or raise ``OfflineCacheMiss`` in offline mode).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bump this integer whenever the on-disk blob format changes in a breaking way.
CACHE_SCHEMA_VERSION: int = 1

# Characters that are unsafe in Windows filenames (tickers can contain '.', '^').
_SAFE = str.maketrans(dict.fromkeys('<>:"/\\|?*', "_"))

# Windows reserved DEVICE names: a file whose stem (the part before the first
# '.') is one of these is the device, NOT a file -- even WITH an extension. e.g.
# "CON.XETRA.json" resolves to the console device, so reading it blocks forever
# waiting on console input. Continental's ticker is CON.XETRA, which lands here.
# We prefix such names with '_' so they become ordinary files. Deterministic and
# applied identically on read and write, so the cache stays self-consistent.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class OfflineCacheMiss(RuntimeError):
    """Raised in offline mode when a requested item is not in the snapshot."""


class RawCache:
    def __init__(self, cache_dir: Path, as_of: str, *, offline: bool = False):
        self.root = Path(cache_dir) / as_of
        self.as_of = as_of
        # Offline mode NEVER touches the network: it reads only the frozen
        # snapshot, so results are byte-identical to anyone with the same
        # snapshot. A miss is a hard error (the snapshot is incomplete) rather
        # than a silent live fetch that could drift.
        self.offline = offline

    def _path(self, endpoint: str, symbol: str) -> Path:
        safe = symbol.translate(_SAFE)
        if safe.split(".", 1)[0].upper() in _RESERVED:
            safe = "_" + safe
        return self.root / endpoint / f"{safe}.json"

    @staticmethod
    def _meta_path(blob_path: Path) -> Path:
        """Return the sidecar path for a given blob path.

        ``AAPL_US.json`` → ``AAPL_US.meta.json`` (same directory).
        """
        return blob_path.with_name(blob_path.stem + ".meta.json")

    @staticmethod
    def _write_meta(meta_path: Path, endpoint: str) -> None:
        """Atomically write the schema sidecar next to the blob."""
        meta = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "format": endpoint,
        }
        payload = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        # Write to a sibling temp file in the same directory so os.replace is
        # guaranteed to be atomic on the same filesystem.
        dir_ = meta_path.parent
        fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".meta-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, meta_path)
        except Exception:
            # Clean up the temp file if anything went wrong before the replace.
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @staticmethod
    def _schema_valid(meta_path: Path) -> bool:
        """
        Return True when the blob should be treated as a cache hit.

        * Sidecar absent → True.  Legacy blobs (written before schema versioning
          was introduced) are treated as current so we don't nuke the existing
          4 GB cache on the first upgrade.  Only an explicit schema_version
          mismatch triggers a re-fetch.
        * Sidecar present, version matches → True.
        * Sidecar present, version mismatch → False (stale format, must re-fetch).
        """
        if not meta_path.exists():
            # Legacy cache file -- absent sidecar means "assume current version".
            return True
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt sidecar -- treat as a miss so we re-fetch cleanly.
            return False
        return int(meta.get("schema_version", -1)) == CACHE_SCHEMA_VERSION

    def get_or_fetch(
        self, endpoint: str, symbol: str, fetch: Callable[[], Any]
    ) -> Any:
        """Return cached JSON for (endpoint, symbol) or fetch, store, and return it."""
        path = self._path(endpoint, symbol)
        meta_path = self._meta_path(path)
        if path.exists() and self._schema_valid(meta_path):
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise OfflineCacheMiss(
                f"offline: {endpoint}/{symbol} not in snapshot {self.as_of}. "
                "Restore the matching data release, or rebuild the snapshot online."
            )
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys for a stable, diff-able, reproducible on-disk representation.
        path.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        self._write_meta(meta_path, endpoint)
        return data

    def exists(self, endpoint: str, symbol: str) -> bool:
        return self._path(endpoint, symbol).exists()
