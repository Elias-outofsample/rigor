"""Quick check that your EODHD key works and the plan/add-on are sufficient.

Run from the repo root after entering your key in `.env` (see docs/data.md):

    python verify_eodhd.py

It prints your plan, a sample price (raw + adjusted), the S&P 500 historical-
constituents count, and a delisted-name price depth - then a PASS/ATTENTION line.
Read-only; makes a handful of small requests. Exits non-zero if the key is unusable.
"""
from __future__ import annotations

import sys

try:
    from rigor.data import DataConfig, DataLoader
    from rigor.data.config import get_api_key
except Exception as exc:  # import/install problem, not a key problem
    print(f"ERROR: could not import rigor ({exc}).")
    print("Install first:  pip install -e .")
    sys.exit(2)


def main() -> int:
    # 1. key present?
    try:
        get_api_key()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    data = DataLoader(DataConfig.from_env())  # as-of defaults to today
    client = data.client
    ok = True

    # 2. plan / subscription
    try:
        user = client._get("user")
        if isinstance(user, dict) and user:
            plan = user.get("subscriptionType") or user.get("name") or "(unknown)"
            print(f"[OK]  plan: {plan}")
        else:
            print("[--]  plan: (could not read /user - not fatal)")
    except Exception as exc:
        print(f"[--]  plan: (could not read /user: {exc} - not fatal)")

    # 3. sample EOD price (raw + adjusted)
    try:
        p = data.prices("AAPL")
        last = p.iloc[-1]
        print(f"[OK]  AAPL latest {last['date']}: close={last['close']:.2f} "
              f"adjusted={last['adjusted_close']:.2f}  ({len(p)} rows)")
    except Exception as exc:
        print(f"[FAIL] AAPL prices failed: {exc}")
        ok = False

    # 4. S&P 500 historical constituents (the add-on)
    try:
        hc = client.historical_constituents("GSPC.INDX")
        n = len(hc) if isinstance(hc, dict) else 0
        if n:
            print(f"[OK]  S&P 500 historical constituents: {n} membership records")
        else:
            print("[FAIL] S&P 500 historical constituents: EMPTY - the Historical "
                  "Constituents add-on may be missing from the plan")
            ok = False
    except Exception as exc:
        print(f"[FAIL] historical constituents failed: {exc}")
        ok = False

    # 5. a delisted name has price history (survivorship-free coverage)
    try:
        d = client.eod("SIVB.US")
        if d:
            print(f"[OK]  delisted-name depth: SIVB.US has {len(d)} bars")
        else:
            print("[--]  delisted-name depth: SIVB.US returned no bars (check plan delisted access)")
    except Exception as exc:
        print(f"[--]  delisted-name depth: {exc}")

    print()
    if ok:
        print("PASS - your key works and the plan covers what the framework needs.")
        return 0
    print("ATTENTION - something above failed; see the [FAIL] lines (often a plan/add-on gap).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
