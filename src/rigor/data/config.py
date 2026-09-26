"""Data-layer configuration: paths, the .env loader, and the as-of convention.

The whole reproducibility story rests on two ideas that live here:

  * A single canonical data source (EODHD) with the key read from the local
    ``.env`` file -- never hard-coded, never committed.
  * An explicit *as-of date*. Every pull is bounded by the as-of date and the
    cache is keyed by it, so two machines that pin the same as-of date get
    byte-identical inputs regardless of *when* they actually ran the pull.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


# src/rigor/data/config.py -> parents[2] == src/ (fallback when no .git is found)
def find_repo_root() -> Path:
    """Locate the workspace root (where .env / data_cache live).

    Walks up from the current working directory and from this file looking for a
    ``.git`` directory. This keeps ``.env`` and ``data_cache/`` anchored at the
    repo root even though the ``rigor`` package lives under ``src/``.
    Falls back to the package's grandparent, then the CWD.
    """
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        p = base
        while True:
            if (p / ".git").exists():
                return p
            if p.parent == p:
                break
            p = p.parent
    return Path(__file__).resolve().parents[2]


REPO_ROOT = find_repo_root()
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache"

EODHD_BASE_URL = "https://eodhd.com/api"
DEFAULT_EXCHANGE = "US"


def load_env(env_path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. No external dependency.

    Ignores blank lines and ``#`` comments; strips surrounding quotes/space.
    """
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def get_api_key(env_path: Path = ENV_PATH) -> str:
    """Return the EODHD API key from .env or raise a clear, actionable error."""
    key = load_env(env_path).get("EODHD_API_KEY", "")
    if not key or key == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            f"EODHD_API_KEY is not set. Open {env_path} and enter your key "
            "(see docs/data.md)."
        )
    return key


def get_fred_api_key(env_path: Path = ENV_PATH) -> str:
    """Optional free FRED API key. When set, FRED series are fetched from the official
    api.stlouisfed.org API (reliable) instead of the flaky public graph CSV endpoint.
    Returns "" if unset -- the framework still works, just less reliably for FRED."""
    key = load_env(env_path).get("FRED_API_KEY", "")
    return "" if key == "PASTE_YOUR_FRED_KEY_HERE" else key


@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for the data layer.

    ``as_of`` is the linchpin of reproducibility. Pin it (e.g. "2026-06-01")
    for a research cycle so every machine pulls the same bounded history. Left
    as ``None`` it defaults to today -- convenient for ad-hoc use, but NOT
    reproducible across days, so production runs should always set it.
    """

    api_key: str
    fred_api_key: str = ""   # optional; when set, use the official FRED API
    base_url: str = EODHD_BASE_URL
    cache_dir: Path = DEFAULT_CACHE_DIR
    exchange: str = DEFAULT_EXCHANGE
    as_of: str | None = None
    offline: bool = False   # read only the frozen snapshot; never fetch live
    timeout: int = 30
    max_retries: int = 3
    # 100k calls/day on the All-in-One plan -> no throttling needed by default.
    min_request_interval_s: float = 0.0

    @property
    def as_of_date(self) -> str:
        """Resolved as-of date string (YYYY-MM-DD); today's date if unset."""
        return self.as_of or date.today().isoformat()

    @classmethod
    def from_env(cls, *, as_of: str | None = None, **overrides) -> DataConfig:
        """Build a config with the key loaded from .env.

        Offline runs read only the frozen snapshot, so no key is needed.
        """
        offline = overrides.get("offline")
        key = "OFFLINE" if offline else get_api_key()
        overrides.setdefault("fred_api_key", "" if offline else get_fred_api_key())
        return cls(api_key=key, as_of=as_of, **overrides)
