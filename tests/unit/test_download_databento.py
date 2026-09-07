"""download_databento.py: the timezone conversion is the single
riskiest part of this script (see its docstring -- an earlier version
of this project's own pipeline had a real, consequential bug from
exactly this kind of mismatch). This round-trips a known UTC instant
through _row_line() and back through the EXISTING, already-audited
loader (load_bars_csv / _parse_yyyymmdd_hhmmss) and asserts the
original UTC timestamp is recovered exactly.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "download_databento", ROOT / "scripts" / "download_databento.py"
)
download_databento = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_databento)

from topstep50k.data.loaders import detect_schema


def _round_trip(ts_utc: datetime, tmp_path: Path) -> datetime:
    line = download_databento._row_line(ts_utc, 100.0, 101.0, 99.0, 100.5, 42)
    p = tmp_path / "roundtrip.txt"
    p.write_text(line)
    _, splitter, row_parser = detect_schema(p)
    bar = row_parser(splitter(line))
    assert bar is not None
    return bar.ts


def test_round_trip_winter_est(tmp_path):
    # 2026-01-15 14:30:00 UTC = 09:30:00 EST (UTC-5, no DST)
    ts_utc = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    recovered = _round_trip(ts_utc, tmp_path)
    assert recovered == ts_utc


def test_round_trip_summer_edt(tmp_path):
    # 2026-07-15 13:30:00 UTC = 09:30:00 EDT (UTC-4, DST active)
    ts_utc = datetime(2026, 7, 15, 13, 30, 0, tzinfo=timezone.utc)
    recovered = _round_trip(ts_utc, tmp_path)
    assert recovered == ts_utc


def test_row_line_local_wallclock_matches_expected_offset():
    # Winter: 14:30 UTC -> 09:30 local (EST, -5h)
    ts_utc = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    line = download_databento._row_line(ts_utc, 1.0, 1.0, 1.0, 1.0, 1)
    assert line.startswith("20260115 093000;")

    # Summer: 13:30 UTC -> 09:30 local (EDT, -4h)
    ts_utc_summer = datetime(2026, 7, 15, 13, 30, 0, tzinfo=timezone.utc)
    line_summer = download_databento._row_line(ts_utc_summer, 1.0, 1.0, 1.0, 1.0, 1)
    assert line_summer.startswith("20260715 093000;")


def test_price_formatting_matches_existing_file_convention():
    ts_utc = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    line = download_databento._row_line(ts_utc, 1061.25, 1061.75, 1061.25, 1061.25, 206)
    # Matches the observed convention in data/raw/es_databento.txt:
    # "20100606 200000;1061.25000000;1061.75000000;1061.25000000;1061.25000000;206"
    assert "1061.25000000" in line
    assert line.rstrip("\n").endswith(";206")
