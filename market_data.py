"""Load Databento 1-minute futures bars and cut them into full futures sessions.

Session definition (US Eastern clock, the clock the raw files are in):
  * A session starts at the 18:00 ET futures open (the previous evening) and is traded until
    16:00 ET. The 16:00-18:00 ET window is excluded: Topstep wants positions flat by 16:10 ET
    (3:10 PM CT) and the exchange halts 17:00-18:00 ET.
  * The session is labelled with the calendar date of its 16:00 ET end (Sunday evening belongs to Monday).
  * VWAP is anchored at the 18:00 ET open.

Time handling for automation:
  * Raw timestamps are naive Eastern wall-clock times; they are localised to America/New_York once.
  * Everything downstream uses `minute_of_session` (0 = 18:00 ET), never a fixed UTC offset or the
    host's local zone. A bot in Asia/Phnom_Penh (UTC+7, no DST) must schedule from the tz-aware
    session open: 18:00 ET = 05:00 or 06:00 ICT depending on US daylight saving.
  * Chicago and Eastern are the same instants (CT = ET - 1h, both shift together), so using the
    Eastern clock does not change which bars are traded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SOURCE_TZ = ZoneInfo("America/New_York")
OPERATOR_TZ = ZoneInfo("Asia/Phnom_Penh")  # display / scheduling only, never used in logic

RAW_DIR = Path(os.environ.get("TOPSTEP_RAW_DIR", "/home/user/data_raw"))

# contract multiplier ($/point) and tick size for the full-size contracts
SPECS: Dict[str, Dict[str, float]] = {
    "es": {"mult": 50.0, "tick": 0.25},
    "nq": {"mult": 20.0, "tick": 0.25},
    "gc": {"mult": 100.0, "tick": 0.10},
}

SESSION_OPEN_MIN = 18 * 60       # 18:00 ET
SESSION_END_MIN = 16 * 60        # last tradable minute is 15:59 ET
MIN_BARS_PER_SESSION = 500
# Roll detection (continuous-contract splices show up as one large non-reverting 1-minute jump)
ROLL_MIN_REL_JUMP = 0.0008       # a splice smaller than 8bp is treated as no roll
ROLL_MONTHS = {"es": (3, 6, 9, 12), "nq": (3, 6, 9, 12), "gc": tuple(range(1, 13))}


def load_minute(market: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return 1-minute OHLCV indexed by tz-aware America/New_York timestamps."""
    pkl = raw_dir / f"{market}_1m.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(raw_dir / f"{market}_databento.txt", sep=";", header=None,
                         names=["ts", "o", "h", "l", "c", "v"], dtype={"ts": str})
        df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
        df = df.set_index("ts")
    # Markets are closed when US clocks change (Sunday 02:00), so nothing is ambiguous or missing.
    df = df.copy()
    df.index = df.index.tz_localize(SOURCE_TZ, ambiguous="raise", nonexistent="raise")
    return df


@dataclass
class Sessions:
    """Full futures sessions (18:00 ET -> 16:00 ET) stored as flat arrays plus offsets."""
    market: str
    dates: np.ndarray           # datetime64[D] session label
    offsets: np.ndarray         # start of each session in the flat arrays (len n+1)
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    smin: np.ndarray            # minutes since the 18:00 ET open (0..1319)
    roll: np.ndarray            # bool per session: contains a suspected contract roll -> skip
    mult: float
    tick: float

    def __len__(self) -> int:
        return len(self.dates)

    def day(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return (self.o[a:b], self.h[a:b], self.l[a:b], self.c[a:b], self.v[a:b], self.smin[a:b])


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def _flag_rolls(market: str, c: np.ndarray, offsets: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Flag one session per expiry window: the one holding the largest 1-minute jump.

    Index futures (ES/NQ): window = the 10 days before each quarterly third Friday.
    Gold (GC): a window around every month end (the vendor's front-month roll can be monthly).
    A continuous-contract splice is a single non-reverting jump, so the biggest jump in the
    window is the roll. Over-flagging only costs a few skipped sessions.
    """
    idx = pd.DatetimeIndex(dates)
    n = len(dates)
    max_rel = np.zeros(n)
    for i in range(n):
        x = c[offsets[i]:offsets[i + 1]]
        if x.size > 50:
            max_rel[i] = (np.abs(np.diff(x)) / x[1:]).max()
    flag = np.zeros(n, dtype=bool)
    windows = []
    for year in range(idx.year.min(), idx.year.max() + 1):
        for month in ROLL_MONTHS[market]:
            if market == "gc":
                anchor = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                windows.append((anchor - pd.Timedelta(days=14), anchor + pd.Timedelta(days=3)))
            else:
                anchor = _third_friday(year, month)
                windows.append((anchor - pd.Timedelta(days=10), anchor))
    for lo, hi in windows:
        sel = np.flatnonzero((idx >= lo) & (idx <= hi))
        if sel.size == 0:
            continue
        k = sel[max_rel[sel].argmax()]
        if max_rel[k] >= ROLL_MIN_REL_JUMP:
            flag[k] = True
    return flag


def build_sessions(market: str, raw_dir: Path = RAW_DIR) -> Sessions:
    cache = raw_dir / f"{market}_sessions_v2.npz"
    spec = SPECS[market]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return Sessions(market, z["dates"].astype("datetime64[D]"), z["offsets"], z["o"], z["h"],
                        z["l"], z["c"], z["v"], z["smin"], z["roll"], spec["mult"], spec["tick"])
    df = load_minute(market, raw_dir)
    mod = (df.index.hour * 60 + df.index.minute).to_numpy()
    keep = (mod >= SESSION_OPEN_MIN) | (mod < SESSION_END_MIN)
    d = df.loc[keep]
    mod = mod[keep]
    wall = d.index.tz_localize(None)                      # Eastern wall clock
    label = (wall + pd.Timedelta(hours=6)).normalize()    # 18:00 ET -> next calendar date
    label = np.asarray(label.values.astype("datetime64[D]"))
    weekday = pd.DatetimeIndex(label).dayofweek.to_numpy()
    ok_day = weekday < 5
    d, mod, label = d.loc[ok_day], mod[ok_day], label[ok_day]
    change = np.flatnonzero(np.diff(label.astype("int64"))) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(d)]
    good = (ends - starts) >= MIN_BARS_PER_SESSION
    starts, ends = starts[good], ends[good]
    sel = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    offsets = np.r_[0, np.cumsum(ends - starts)]
    arrs = {k: d[k].to_numpy(dtype=np.float64)[sel] for k in ("o", "h", "l", "c", "v")}
    smin = ((mod[sel] - SESSION_OPEN_MIN) % 1440).astype(np.int32)
    dates = label[starts]
    roll = _flag_rolls(market, arrs["c"], offsets, dates)
    np.savez(cache, dates=dates.astype("datetime64[D]").astype("int64"), offsets=offsets, smin=smin,
             roll=roll, **arrs)
    return Sessions(market, dates, offsets, arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["v"],
                    smin, roll, spec["mult"], spec["tick"])


def session_open_phnom_penh(session_label: datetime) -> datetime:
    """When a session opens in Phnom Penh time (display / scheduling helper).

    `session_label` is the session's date; the open is 18:00 ET on the previous calendar day.
    """
    prev = session_label.replace(hour=18, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
    prev = prev.replace(tzinfo=SOURCE_TZ)
    return prev.astimezone(OPERATOR_TZ)
