"""Build the 'live-equivalent' NQ session cache: a continuous, back-adjusted series in which EVERY session is traded.

Why (audit, October 2026): the Databento continuous NQ series is UNADJUSTED - at each quarterly roll the series jumps by the calendar spread (the first bar after 00:00 UTC, i.e. 19:00 or 20:00 ET).
market_data._flag_rolls flagged, in each roll window, the session holding the largest 1-minute jump; that is the true splice session in only 22 of 65 quarters (43 of the 51 quarters before 2023 are
wrong), so the project skipped ~42 ex-post volatile days (a hindsight filter) and traded the 42 real splice sessions with the spread inside the session or in the overnight gap. This script
(1) finds the true splice per quarter (the largest jump at the first bar after 00:00 UTC inside [third Friday - 16d, third Friday + 2d]),
(2) back-adjusts every earlier bar by the jump (difference adjustment, cumulative), so open/close/gap are contract-consistent, and
(3) clears the roll flags so no session is skipped. Run it once; point TOPSTEP_RAW_DIR at the output to re-run any study on it.

    python repair_splices.py            # writes /home/user/data_raw_live/nq_sessions_v2.npz (es/gc symlinked) and results/splices_nq.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import RAW_DIR, _third_friday, load_minute

OUT = Path("/home/user/data_raw_live")


def find_splices(df: pd.DataFrame, jump_min_bp: float = 0.0) -> pd.DataFrame:
    idx = df.index
    utc = idx.tz_convert("UTC")
    d = utc.normalize()
    first = np.r_[False, d.values[1:] != d.values[:-1]]
    c, o = df["c"].to_numpy(), df["o"].to_numpy()
    jump = np.r_[np.nan, o[1:] - c[:-1]]
    gap_min = np.r_[np.nan, (utc[1:] - utc[:-1]).total_seconds() / 60]
    bp = jump / np.r_[np.nan, c[:-1]] * 1e4
    M = pd.DataFrame({"jump": jump, "bp": bp, "gap": gap_min, "h": utc.hour, "m": utc.minute}, index=idx)[first]
    M = M[(M.gap <= 10) & (M.h == 0) & (M.m <= 10)]
    rows = []
    for y in range(idx.year.min(), idx.year.max() + 1):
        for mth in (3, 6, 9, 12):
            tf = _third_friday(y, mth).tz_localize(idx.tz)
            w = M[tf - pd.Timedelta(days=16): tf + pd.Timedelta(days=2)]
            if len(w) == 0:
                continue
            k = w.bp.abs().idxmax()
            rows.append(dict(expiry=tf.date(), ts=k, jump_pts=float(w.jump[k]), bp=float(w.bp[k]), second_bp=float(w.bp.abs().sort_values(ascending=False).iloc[1]) if len(w) > 1 else np.nan,
                             session=(k.tz_localize(None) + pd.Timedelta(hours=6)).normalize().date()))
    return pd.DataFrame(rows)


def main() -> None:
    df = load_minute("nq")
    T = find_splices(df)
    print(f"{len(T)} quarterly splices found; |jump| median {T.jump_pts.abs().median():.2f} pts, max {T.jump_pts.abs().max():.1f} pts")
    z = np.load(RAW_DIR / "nq_sessions_v2.npz", allow_pickle=False)
    d = {k: z[k].copy() for k in z.files}
    dates = pd.DatetimeIndex(d["dates"].astype("datetime64[D]"))
    off, smin = d["offsets"], d["smin"]
    flat_idx, jumps = [], []
    for r in T.itertuples():
        j = int(np.searchsorted(dates.values, np.datetime64(r.session)))
        if j >= len(dates) or dates[j] != pd.Timestamp(r.session):
            print("splice session not in the session set:", r.session)
            continue
        k_smin = ((r.ts.hour * 60 + r.ts.minute) - 18 * 60) % 1440
        a, b = off[j], off[j + 1]
        k = a + int(np.searchsorted(smin[a:b], k_smin))
        flat_idx.append(k)
        jumps.append(r.jump_pts)
    order = np.argsort(flat_idx)
    flat_idx, jumps = np.array(flat_idx)[order], np.array(jumps)[order]
    # cumulative difference back-adjustment: every bar before splice i is shifted by the sum of the jumps of splice i and all LATER splices
    adj = np.zeros(len(d["o"]))
    for k, jmp in zip(flat_idx, jumps):
        adj[:k] += jmp
    for key in ("o", "h", "l", "c"):
        d[key] = d[key] + adj
    d["roll"] = np.zeros(len(dates), dtype=bool)
    OUT.mkdir(exist_ok=True)
    for m in ("es", "gc"):
        t = OUT / f"{m}_sessions_v2.npz"
        if t.is_symlink() or t.exists():
            t.unlink()
        t.symlink_to(RAW_DIR / f"{m}_sessions_v2.npz")
    np.savez(OUT / "nq_sessions_v2.npz", **d)
    T.to_csv("results/splices_nq.csv", index=False)
    # check: no splice jump left at the splice bars
    resid = [float(d["o"][k] - d["c"][k - 1]) for k in flat_idx]
    print("residual open-vs-previous-close at the splice bars: max |x| =", round(max(abs(x) for x in resid), 2), "pts (a normal minute move)")
    print(f"wrote {OUT / 'nq_sessions_v2.npz'}; sessions {len(dates)}, roll flags cleared")


if __name__ == "__main__":
    main()
