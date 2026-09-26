"""Test-session setup.

Licensed point-in-time membership files are not shipped; the suite runs against a
synthetic one covering the fake universe of ``fake_data.FakeDataLoader``.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "RIGOR_REFERENCE_DIR", str(Path(__file__).resolve().parent / "fixtures" / "reference")
)
