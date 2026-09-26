"""Model registry — append-only JSONL log of every strategy run.

Persisted at ``catalog/model_registry.jsonl`` (relative to the repo root).
Each line is a JSON object representing one ``ModelRecord``.

Public API
----------
record(entry, *, path)            -> ModelRecord  — append one record
load(path)                        -> list[ModelRecord]
latest(slug, path)                -> ModelRecord | None
lineage(model_id, path)           -> list[ModelRecord]  (root → … → given)
query(path, *, slug, category, status)  -> list[ModelRecord]
to_sqlite(path)                   -> sqlite3.Connection (in-memory)
record_strategy_run(strategy_dir) -> ModelRecord  — read config.json + summary.json

CLI
---
python -m rigor.project.registry list
python -m rigor.project.registry show <model_id>
python -m rigor.project.registry record <strategy_dir>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers — repo root & default path
# ---------------------------------------------------------------------------

def _default_path() -> Path:
    # Reuse the canonical repo-root finder (rigor.data.config) instead of a
    # second hand-rolled walker — single source of truth, no drift.
    from rigor.data.config import find_repo_root
    return find_repo_root() / "catalog" / "model_registry.jsonl"


# ---------------------------------------------------------------------------
# ModelRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelRecord:
    """One entry in the model registry."""

    model_id: str
    slug: str
    category: str
    version: str
    params: dict
    metrics: dict
    dataset_hash: str | None
    code_fingerprint: str | None
    config_fingerprint: str | None
    parent_model_id: str | None
    status: str
    note: str
    created_at: str  # ISO-8601 UTC

    # Allow extra keys from future versions without breaking old readers.
    _extra: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "slug": self.slug,
            "category": self.category,
            "version": self.version,
            "params": self.params,
            "metrics": self.metrics,
            "dataset_hash": self.dataset_hash,
            "code_fingerprint": self.code_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "parent_model_id": self.parent_model_id,
            "status": self.status,
            "note": self.note,
            "created_at": self.created_at,
        }
        d.update(self._extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelRecord:
        """Deserialise from a plain dict; unknown keys land in ``_extra``."""
        known = {
            "model_id", "slug", "category", "version", "params", "metrics",
            "dataset_hash", "code_fingerprint", "config_fingerprint",
            "parent_model_id", "status", "note", "created_at",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            model_id=d.get("model_id", ""),
            slug=d.get("slug", ""),
            category=d.get("category", ""),
            version=d.get("version", "v1"),
            params=d.get("params") or {},
            metrics=d.get("metrics") or {},
            dataset_hash=d.get("dataset_hash"),
            code_fingerprint=d.get("code_fingerprint"),
            config_fingerprint=d.get("config_fingerprint"),
            parent_model_id=d.get("parent_model_id"),
            status=d.get("status", ""),
            note=d.get("note", ""),
            created_at=d.get("created_at", ""),
            _extra=extra,
        )


# ---------------------------------------------------------------------------
# Fingerprinting & ID generation
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of the given file's bytes."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fingerprint_strategy(strategy_dir: Path | str) -> dict[str, str | None]:
    """Return ``{"code_fingerprint": ..., "config_fingerprint": ...}``.

    Both values are SHA-256 hex strings when the corresponding file exists,
    otherwise ``None``.
    """
    d = Path(strategy_dir)
    code_fp: str | None = None
    config_fp: str | None = None

    strategy_py = d / "strategy.py"
    if strategy_py.exists():
        code_fp = _sha256_file(strategy_py)

    config_json = d / "config.json"
    if config_json.exists():
        config_fp = _sha256_file(config_json)

    return {"code_fingerprint": code_fp, "config_fingerprint": config_fp}


def make_model_id(
    slug: str, version: str, params: dict, parent_model_id: str | None = None
) -> str:
    """First 12 hex chars of SHA-256 over canonical JSON of
    (slug, version, params, parent_model_id).

    The id is the *identity* of a model — a deterministic function of its config
    and its place in the lineage, so the same model always hashes the same.
    Wall-clock ``created_at`` is deliberately excluded (it lives on the record)
    so recording a strategy is reproducible across runs instead of minting a fresh
    id every time; ``parent_model_id`` keeps successive lineage nodes distinct even
    when their params are unchanged.
    """
    payload = json.dumps(
        {"slug": slug, "version": version, "params": params,
         "parent_model_id": parent_model_id},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _nan_safe_loads(text: str) -> Any:
    """Parse JSON, replacing bare ``NaN`` literals with ``null`` first."""
    cleaned = re.sub(r"\bNaN\b", "null", text)
    return json.loads(cleaned)


def record(entry: ModelRecord | dict, *, path: Path | None = None) -> ModelRecord:
    """Append *entry* to the JSONL registry and return the ``ModelRecord``.

    Creates the parent directory and file on first use.
    """
    rec = ModelRecord.from_dict(entry) if isinstance(entry, dict) else entry

    target = path if path is not None else _default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec.to_dict(), separators=(",", ":")) + "\n")

    return rec


def load(path: Path | None = None) -> list[ModelRecord]:
    """Read all ``ModelRecord`` objects from the JSONL file.

    Malformed lines are silently skipped so a single corrupt entry never
    prevents reading the rest of the log.
    """
    target = path if path is not None else _default_path()
    if not target.exists():
        return []

    records: list[ModelRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        with contextlib.suppress(Exception):
            records.append(ModelRecord.from_dict(_nan_safe_loads(line)))
    return records


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def latest(slug: str, path: Path | None = None) -> ModelRecord | None:
    """Return the most recently created record for *slug*, or ``None``."""
    matches = [r for r in load(path) if r.slug == slug]
    if not matches:
        return None
    return max(matches, key=lambda r: r.created_at)


def lineage(model_id: str, path: Path | None = None) -> list[ModelRecord]:
    """Trace the parent chain and return records ordered root → … → *model_id*.

    Returns ``[]`` if *model_id* is not found.
    """
    all_records = load(path)
    by_id: dict[str, ModelRecord] = {r.model_id: r for r in all_records}

    target_rec = by_id.get(model_id)
    if target_rec is None:
        return []

    chain: list[ModelRecord] = []
    seen: set[str] = set()
    current: ModelRecord | None = target_rec
    while current is not None:
        if current.model_id in seen:
            break  # guard against cycles
        seen.add(current.model_id)
        chain.append(current)
        pid = current.parent_model_id
        current = by_id.get(pid) if pid else None  # type: ignore[arg-type]

    chain.reverse()
    return chain


def query(
    path: Path | None = None,
    *,
    slug: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> list[ModelRecord]:
    """Return records filtered by any combination of slug / category / status."""
    results = load(path)
    if slug is not None:
        results = [r for r in results if r.slug == slug]
    if category is not None:
        results = [r for r in results if r.category == category]
    if status is not None:
        results = [r for r in results if r.status == status]
    return results


# ---------------------------------------------------------------------------
# SQLite export
# ---------------------------------------------------------------------------

def to_sqlite(path: Path | None = None) -> sqlite3.Connection:
    """Load all records into an in-memory SQLite database and return the connection.

    The ``model_registry`` table has one column per ``ModelRecord`` field;
    ``params`` and ``metrics`` are stored as JSON text.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE model_registry (
            model_id          TEXT PRIMARY KEY,
            slug              TEXT NOT NULL,
            category          TEXT NOT NULL,
            version           TEXT NOT NULL,
            params            TEXT NOT NULL,
            metrics           TEXT NOT NULL,
            dataset_hash      TEXT,
            code_fingerprint  TEXT,
            config_fingerprint TEXT,
            parent_model_id   TEXT,
            status            TEXT NOT NULL,
            note              TEXT NOT NULL,
            created_at        TEXT NOT NULL
        )
    """)
    rows = [
        (
            r.model_id, r.slug, r.category, r.version,
            json.dumps(r.params, separators=(",", ":")),
            json.dumps(r.metrics, separators=(",", ":")),
            r.dataset_hash, r.code_fingerprint, r.config_fingerprint,
            r.parent_model_id, r.status, r.note, r.created_at,
        )
        for r in load(path)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO model_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Convenience: record a full strategy run
# ---------------------------------------------------------------------------

def record_strategy_run(
    strategy_dir: Path | str,
    *,
    version: str | None = None,
    params: dict | None = None,
    parent_model_id: str | None = None,
    status: str | None = None,
    note: str = "",
    path: Path | None = None,
) -> ModelRecord:
    """Read ``config.json`` + ``artifacts/<slug>_summary.json``, build a
    ``ModelRecord``, append it to the registry, and return it.

    Parameters
    ----------
    strategy_dir:
        Path to the strategy folder (must contain ``config.json``).
    version:
        Override the version from ``config.json`` (defaults to ``config["version"]``
        or ``"v1"``).
    params:
        Override the params dict (defaults to ``config["extra"]`` or ``{}``).
    parent_model_id:
        If this run continues or derives from a previous model, pass its ID here.
    status:
        Override the deployment status (defaults to ``config["status"]`` or ``""``).
    note:
        Free-text annotation for this record.
    path:
        Override the default registry JSONL path.
    """
    d = Path(strategy_dir)

    # --- config.json --------------------------------------------------------
    config_path = d / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {d}")
    raw_config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    slug: str = raw_config.get("slug") or d.name
    category: str = raw_config.get("category", "other")
    cfg_version: str = version or str(raw_config.get("version", "v1"))
    cfg_status: str = status or str(raw_config.get("status", ""))
    cfg_params: dict = params if params is not None else (raw_config.get("extra") or {})

    # --- artifacts/<slug>_summary.json --------------------------------------
    summary_path = d / "artifacts" / f"{slug}_summary.json"
    metrics: dict = {}
    if summary_path.exists():
        raw_summary = _nan_safe_loads(summary_path.read_text(encoding="utf-8"))
        # Support both flat {"sharpe": ...} and nested {"metrics": {"sharpe": ...}}
        if "metrics" in raw_summary and isinstance(raw_summary["metrics"], dict):
            metrics = raw_summary["metrics"]
        else:
            metrics = {k: v for k, v in raw_summary.items() if k != "verdict"}

    # --- fingerprints -------------------------------------------------------
    fps = fingerprint_strategy(d)

    # --- build record -------------------------------------------------------
    created_at = datetime.now(tz=UTC).isoformat()
    model_id = make_model_id(slug, cfg_version, cfg_params, parent_model_id)

    rec = ModelRecord(
        model_id=model_id,
        slug=slug,
        category=category,
        version=cfg_version,
        params=cfg_params,
        metrics=metrics,
        dataset_hash=None,
        code_fingerprint=fps["code_fingerprint"],
        config_fingerprint=fps["config_fingerprint"],
        parent_model_id=parent_model_id,
        status=cfg_status,
        note=note,
        created_at=created_at,
    )

    return record(rec, path=path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: ``python -m rigor.project.registry <subcommand>``."""
    from ..console import force_utf8_stdio
    force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="rigor-registry",
        description="Rigor model registry — append-only JSONL log of strategy runs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    list_p = sub.add_parser("list", help="List all registry entries.")
    list_p.add_argument("--slug", default=None, help="Filter by slug.")
    list_p.add_argument("--category", default=None, help="Filter by category.")
    list_p.add_argument("--status", default=None, help="Filter by status.")
    list_p.add_argument("--path", default=None, type=Path, help="Override registry path.")

    # show
    show_p = sub.add_parser("show", help="Show details for a model_id.")
    show_p.add_argument("model_id", help="The model ID to show.")
    show_p.add_argument("--path", default=None, type=Path, help="Override registry path.")

    # record
    rec_p = sub.add_parser("record", help="Record a strategy run.")
    rec_p.add_argument("strategy_dir", type=Path, help="Path to the strategy folder.")
    rec_p.add_argument("--version", default=None, help="Override version.")
    rec_p.add_argument("--parent", default=None, dest="parent_model_id",
                       help="Parent model ID.")
    rec_p.add_argument("--status", default=None, help="Override status.")
    rec_p.add_argument("--note", default="", help="Free-text note.")
    rec_p.add_argument("--path", default=None, type=Path, help="Override registry path.")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        recs = query(
            args.path,
            slug=args.slug,
            category=args.category,
            status=args.status,
        )
        if not recs:
            print("(no records)")
        for r in recs:
            sharpe = r.metrics.get("sharpe", "-")
            print(f"{r.created_at[:19]}  {r.model_id}  {r.slug:<32}  "
                  f"v={r.version}  status={r.status}  sharpe={sharpe}")

    elif args.cmd == "show":
        recs = load(args.path)
        matches = [r for r in recs if r.model_id == args.model_id]
        if not matches:
            print(f"model_id {args.model_id!r} not found")
            return 1
        rec = matches[-1]
        print(json.dumps(rec.to_dict(), indent=2))

    elif args.cmd == "record":
        rec = record_strategy_run(
            args.strategy_dir,
            version=args.version,
            parent_model_id=args.parent_model_id,
            status=args.status,
            note=args.note,
            path=args.path,
        )
        print(f"Recorded {rec.model_id}  ({rec.slug}  {rec.version}  {rec.status})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
