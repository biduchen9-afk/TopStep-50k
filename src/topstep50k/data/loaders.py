"""Bar loaders.

The TopStep cleaned files use an unknown-to-us schema until the user
shares them. `detect_schema` sniffs the first few lines of a text file
and returns a parser callable; this lets us validate the loader the
moment the files land without code edits.

Supported sniffs:
* NinjaTrader / Sierra format: `YYYYMMDD HHMMSS;O;H;L;C;V`
* CSV with header containing 'date'/'time'/'datetime' + OHLCV
* Pipe / semicolon / comma / tab separators
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from topstep50k.engine.types import Bar


# (parser_name, splitter_callable, parse_row_callable)
SchemaResult = tuple[str, Callable[[str], list[str]], Callable[[list[str]], Bar | None]]

# NinjaTrader/SierraChart bar exports are stamped in the chart's configured
# session timezone -- never true UTC. Verified empirically against
# es/nq/gc_databento.txt (2026-08): the CME daily maintenance-halt volume
# gap sits at raw-labeled 17:00-18:00 in BOTH January (EST) and July (EDT)
# with zero seasonal drift (true UTC would show it move an hour between
# winter/summer), and raw-labeled 09:30/16:00 show the classic NYSE
# cash-open / market-on-close volume spikes. So the raw clock already
# observes US DST -- it's US/Eastern local time. Tagging it UTC (the old
# behavior) silently shifted every session-time computation downstream
# (ORB/MeanRev RTH windows, OD entry/exit timing, the regime gates' RTH
# open/close) by the EST/EDT offset.
_NINJATRADER_SOURCE_TZ = ZoneInfo("America/New_York")


def _sep_split(sep: str) -> Callable[[str], list[str]]:
    return lambda line: line.rstrip("\n").split(sep)


def _parse_yyyymmdd_hhmmss(parts: list[str]) -> Bar | None:
    # YYYYMMDD HHMMSS;O;H;L;C;V  (NinjaTrader/SierraChart export, stamped
    # in US/Eastern local time -- see _NINJATRADER_SOURCE_TZ above)
    if len(parts) < 6:
        return None
    dt_str = parts[0]
    if " " in dt_str:
        d, t = dt_str.split(" ", 1)
    else:
        d, t = dt_str, "000000"
    naive = datetime.strptime(d + t, "%Y%m%d%H%M%S")
    ts = naive.replace(tzinfo=_NINJATRADER_SOURCE_TZ).astimezone(timezone.utc)
    try:
        o, h, l, c = (float(parts[i]) for i in (1, 2, 3, 4))
        v = int(float(parts[5])) if len(parts) > 5 else 0
    except ValueError:
        return None
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def _parse_iso_dt(parts: list[str]) -> Bar | None:
    # YYYY-MM-DD HH:MM:SS,O,H,L,C,V
    if len(parts) < 5:
        return None
    try:
        ts = datetime.fromisoformat(parts[0].strip())
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        o, h, l, c = (float(parts[i]) for i in (1, 2, 3, 4))
        v = int(float(parts[5])) if len(parts) > 5 else 0
    except ValueError:
        return None
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def detect_schema(path: Path) -> SchemaResult:
    """Sniff the file's separator and date format from its first ~5 lines."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample_lines = []
        for _ in range(8):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line)
    if not sample_lines:
        raise ValueError(f"Empty file: {path}")

    # Pick the separator with the most consistent column counts.
    candidates = [";", ",", "\t", "|"]
    best_sep, best_score = ",", -1
    for sep in candidates:
        counts = [len(line.rstrip("\n").split(sep)) for line in sample_lines if line.strip()]
        if not counts:
            continue
        # Score: prefer higher column count, prefer consistency
        consistent = len(set(counts)) == 1
        score = counts[0] * (10 if consistent else 1)
        if score > best_score:
            best_score = score
            best_sep = sep

    # Skip header if first row's "first column" isn't parseable as a date.
    first_row = sample_lines[0].rstrip("\n").split(best_sep)
    first_field = first_row[0].strip().strip('"').lower()
    has_header = any(k in first_field for k in ("date", "time", "datetime", "timestamp"))

    # Decide date parser by looking at the first DATA row.
    data_row = sample_lines[1 if has_header else 0].rstrip("\n").split(best_sep)
    sample_dt = data_row[0].strip().strip('"')

    if sample_dt[:8].isdigit() and len(sample_dt.split()[0]) == 8:
        parser_name = "yyyymmdd_hhmmss"
        row_parser = _parse_yyyymmdd_hhmmss
    elif "-" in sample_dt[:10]:
        parser_name = "iso"
        row_parser = _parse_iso_dt
    else:
        raise ValueError(f"Unrecognised date format in {path}: {sample_dt!r}")

    return f"{parser_name}/sep={best_sep!r}/header={has_header}", _sep_split(best_sep), row_parser


def load_bars_csv(path: Path) -> Iterator[Bar]:
    """Stream Bars from a cleaned bar file. Skips unparseable rows but
    counts them; caller can compare emitted count to file linecount."""
    path = Path(path)
    schema, splitter, row_parser = detect_schema(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        # Re-open if we used the first line; simpler to just re-read
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if "/header=True" in schema:
            f.readline()
        for line in f:
            if not line.strip():
                continue
            bar = row_parser(splitter(line))
            if bar is not None:
                yield bar


def load_bars_csv_tail(path: Path, tail_bytes: int = 40_000_000) -> Iterator[Bar]:
    """Fast path for loading just the recent end of a large, chronologically-
    sorted bar file (e.g. a multi-year Databento export).

    `load_bars_csv` streams from byte 0, so reading only "the last 200
    days" of a 16-year, ~420MB file still means parsing the entire file
    -- it only skips YIELDING the old rows, not reading them. This seeks
    to `max(0, filesize - tail_bytes)` first and parses from there, which
    is what a daily pre-market signal script actually wants. Default
    tail_bytes (40MB) comfortably covers >200 calendar days of 1-min bars
    for ES/NQ/GC (~7MB/month observed on the Databento files).

    The schema is still sniffed from the FILE'S OWN HEAD (first ~8 lines),
    not the tail, since a mid-file seek can land anywhere relative to a
    header row.
    """
    path = Path(path)
    schema, splitter, row_parser = detect_schema(path)
    has_header = "/header=True" in schema
    file_size = path.stat().st_size
    seek_to = max(0, file_size - tail_bytes)

    with path.open("rb") as f:
        f.seek(seek_to)
        if seek_to > 0:
            f.readline()  # discard the partial line we landed in the middle of
        elif has_header:
            f.readline()  # skip the real header when we didn't seek at all
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            bar = row_parser(splitter(line))
            if bar is not None:
                yield bar


def load_bars_df(path: Path) -> pd.DataFrame:
    """Eager pandas DataFrame loader for ad-hoc analysis. Indexed by ts."""
    path = Path(path)
    rows = list(load_bars_csv(path))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "ts": [b.ts for b in rows],
            "open": [b.open for b in rows],
            "high": [b.high for b in rows],
            "low": [b.low for b in rows],
            "close": [b.close for b in rows],
            "volume": [b.volume for b in rows],
        }
    ).set_index("ts").sort_index()
    return df
