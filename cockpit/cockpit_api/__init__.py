"""cockpit_api — the Portfolio Cockpit web backend (FastAPI).

A small FastAPI facade over ``portfolio_engine`` (the headless engine that ships in
the same package): discover strategies from the Rigor ``strategies/`` book and build /
validate / gate a multi-strategy portfolio. The browser frontend in
``cockpit/frontend`` is the interface; this package is the API it talks to.

Modelled on an earlier in-house cockpit service, but wired to
our engine and strategy book rather than his ``Lib``/``Model``/Norgate stack.
"""
from __future__ import annotations

__version__ = "0.1.0"
