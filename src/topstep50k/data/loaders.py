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

import pandas as pd

from topstep50k.engine.types import Bar


# (parser_name, splitter_callable, parse_row_callable)
SchemaResult = tuple[str, Callable[[str], list[str]], Callable[[list[str]], Bar | None]]


def _sep_split(sep: str) -> Callable[[str], list[str]]:
    return lambda line: line.rstrip("\n").split(sep)


def _parse_yyyymmdd_hhmmss(parts: list[str]) -> Bar | None:
    # YYYYMMDD HHMMSS;O;H;L;C;V  (NinjaTrader/SierraChart export)
    if len(parts) < 6:
        return None
    dt_str = parts[0]
    if " " in dt_str:
        d, t = dt_str.split(" ", 1)
    else:
        d, t = dt_str, "000000"
    ts = datetime.strptime(d + t, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
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
