"""Download + clean new CME futures history from Databento, in the
EXACT on-disk format the rest of this project already expects.

Why this format, not raw UTC
─────────────────────────────
The existing data/raw/{es,nq,gc}_databento.txt files use the
NinjaTrader/SierraChart convention: `YYYYMMDD HHMMSS;O;H;L;C;V` with
the timestamp stamped in US/Eastern LOCAL wall-clock time (see
data/loaders.py's _NINJATRADER_SOURCE_TZ docstring for how that was
verified -- it's the single most consequential bug this project has
had: an earlier version of this pipeline tagged that same convention
as UTC and silently corrupted every session-time computation
downstream). Databento's API returns genuine UTC timestamps. Rather
than teach the loader a second schema (and risk a second timezone
bug), this script converts Databento's UTC timestamps to US/Eastern
local wall-clock strings at DOWNLOAD time, so the output file is
byte-for-byte the same convention as the existing data and the
already-audited loader path (detect_schema / _parse_yyyymmdd_hhmmss)
handles it completely unchanged.

Safety
──────
* Requires DATABENTO_API_KEY in the environment (loaded from .env via
  python-dotenv if present -- see .env.example). Never hardcode a key.
* ALWAYS queries the cost estimate first (client.metadata.get_cost)
  and prints it. Requires --yes to actually spend money and download;
  without it, this only shows the estimate and exits.
* Defaults to the RECENT_START window (2021-12-31 onward) already used
  for most of this project's evaluation scripts, not the full 16-year
  history -- matches what's actually been tested against, and is far
  cheaper than re-pulling 16 years for every new symbol.

Usage
─────
  python scripts/download_databento.py --symbols CL,RTY,6E,ZN --estimate-only
  python scripts/download_databento.py --symbols CL,RTY,6E,ZN --yes
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EASTERN = ZoneInfo("America/New_York")
DEFAULT_START = "2021-12-31"
DEFAULT_END = None  # None = up to now
DATASET = "GLBX.MDP3"  # CME Globex MDP 3.0 -- matches ES/NQ/GC's exchange
SCHEMA = "ohlcv-1m"

# Continuous front-month contract notation (Databento "stype_in=continuous").
# Diversifying beyond ES/NQ/GC (large-cap equity x2 + gold) into genuinely
# different macro drivers: energy, small-cap equity, currency, rates.
DEFAULT_SYMBOL_MAP = {
    "CL": "CL.c.0",   # WTI crude oil -- energy
    "RTY": "RTY.c.0",  # Russell 2000 e-mini -- small-cap equity
    "6E": "6E.c.0",    # Euro FX -- currency
    "ZN": "ZN.c.0",    # 10-year T-note -- rates
}


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY not set. Put it in .env (see .env.example) "
            "or export it before running this script."
        )
    return key


def _fmt_price(x: float) -> str:
    return f"{x:.8f}"


def _row_line(ts_utc: datetime, o: float, h: float, l: float, c: float, v: float) -> str:
    """Convert a UTC timestamp + OHLCV floats into ONE line matching the
    existing NinjaTrader-style convention: local Eastern wall-clock time,
    semicolon-separated, 8-decimal prices, integer-ish volume.

    This is the single riskiest function in the script (timezone
    conversion) -- see test_download_databento.py for the round-trip
    test against the existing loader.
    """
    local = ts_utc.astimezone(EASTERN)
    date_str = local.strftime("%Y%m%d %H%M%S")
    return f"{date_str};{_fmt_price(o)};{_fmt_price(h)};{_fmt_price(l)};{_fmt_price(c)};{int(v)}\n"


def estimate_cost(client, symbol_map: dict[str, str], start: str, end: str | None) -> dict[str, float]:
    costs = {}
    for short, full_symbol in symbol_map.items():
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[full_symbol],
            schema=SCHEMA,
            start=start,
            end=end,
            stype_in="continuous",
        )
        costs[short] = cost
    return costs


def download_symbol(client, short: str, full_symbol: str, start: str, end: str | None,
                     out_path: Path) -> int:
    """Stream one symbol's OHLCV-1m history to `out_path` in the
    project's on-disk convention. Returns the number of rows written.
    """
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[full_symbol],
        schema=SCHEMA,
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    if df.empty:
        return 0
    df = df.sort_index()

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ts, row in df.iterrows():
            ts_utc = ts.to_pydatetime()
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.replace(tzinfo=timezone.utc)
            else:
                ts_utc = ts_utc.astimezone(timezone.utc)
            f.write(_row_line(ts_utc, row["open"], row["high"], row["low"],
                               row["close"], row["volume"]))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOL_MAP.keys()),
                     help="Comma-separated short symbols (must be keys in "
                          "DEFAULT_SYMBOL_MAP, or extend the map for others)")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--estimate-only", action="store_true",
                     help="Print the cost estimate and exit -- no download, no spend.")
    ap.add_argument("--yes", action="store_true",
                     help="Actually spend money and download. Without this, "
                          "the script only estimates cost.")
    args = ap.parse_args()

    import databento as db

    key = _load_api_key()
    client = db.Historical(key)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    symbol_map = {s: DEFAULT_SYMBOL_MAP[s] for s in symbols if s in DEFAULT_SYMBOL_MAP}
    missing = set(symbols) - set(symbol_map)
    if missing:
        raise SystemExit(f"Unknown short symbols (add to DEFAULT_SYMBOL_MAP): {missing}")

    print(f"Dataset={DATASET} schema={SCHEMA} start={args.start} end={args.end or 'now'}")
    print(f"Symbols: {symbol_map}")

    print("\nEstimating cost (no charge for this call)...")
    costs = estimate_cost(client, symbol_map, args.start, args.end)
    total = 0.0
    for short, cost in costs.items():
        print(f"  {short:<4} ({symbol_map[short]}): ${cost:,.2f}")
        total += cost
    print(f"  TOTAL: ${total:,.2f}")

    if args.estimate_only or not args.yes:
        print("\n--yes not passed (or --estimate-only set) -- stopping here, "
              "nothing downloaded, nothing charged.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for short, full_symbol in symbol_map.items():
        out_path = out_dir / f"{short.lower()}_databento.txt"
        print(f"\nDownloading {short} ({full_symbol}) -> {out_path} ...", flush=True)
        n = download_symbol(client, short, full_symbol, args.start, args.end, out_path)
        print(f"  wrote {n:,} rows", flush=True)

    print("\nDone. Verify with: python -c \"from topstep50k.data.loaders import "
          "load_bars_csv; b=list(load_bars_csv('data/raw/cl_databento.txt')); "
          "print(len(b), b[0].ts, b[-1].ts)\"")


if __name__ == "__main__":
    main()
