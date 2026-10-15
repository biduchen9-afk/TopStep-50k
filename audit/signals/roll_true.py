"""Find the true splice: Databento-style continuous series switch at 00:00 UTC => 20:00 ET (EDT) / 19:00 ET (EST).
For each expiry window compare the jump at that clock minute with the project's flagged session."""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from load_raw import load
from market_data import build_sessions, _third_friday

mkt = sys.argv[1] if len(sys.argv) > 1 else 'nq'
df = load(mkt)
s = build_sessions(mkt)
flag = pd.Series(s.roll, index=pd.DatetimeIndex(s.dates))
c = df['c']
prev_c = c.shift(1)
gap_min = (df.index.to_series().diff() / pd.Timedelta(minutes=1))
jump = (df['o'] - prev_c)  # open of this minute vs close of previous minute (splice appears here)
jump = jump.where(gap_min == 1)
ts = df.index
# 00:00 UTC in ET wall clock: DST rule for US (second Sunday March .. first Sunday Nov)
def is_edt(t):
    y = t.year
    mar = pd.Timestamp(y, 3, 1); sun2 = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
    nov = pd.Timestamp(y, 11, 1); sun1 = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
    return (t >= sun2 + pd.Timedelta(hours=2)) & (t < sun1 + pd.Timedelta(hours=2))
rows = []
for y in range(2010, 2027):
    for mth in (3, 6, 9, 12):
        tf = _third_friday(y, mth)
        lo, hi = tf - pd.Timedelta(days=12), tf + pd.Timedelta(days=1)
        w = jump[lo:hi]
        if len(w) == 0:
            continue
        # utc midnight minute
        cand = w[(((w.index.hour == 20) & (w.index.minute == 0)) | ((w.index.hour == 19) & (w.index.minute == 0)))]
        if len(cand) == 0:
            continue
        # choose the time consistent with DST
        cand = cand[[(is_edt(t) and t.hour == 20) or ((not is_edt(t)) and t.hour == 19) for t in cand.index]]
        if len(cand) == 0:
            continue
        k = cand.abs().idxmax()
        sess_date = (k + pd.Timedelta(hours=6)).normalize()
        rel = cand.abs().max() / c[k] * 1e4
        # typical |jump| at the same clock minute on other days
        flagged_dates = flag[flag.index.isin([sess_date])]
        rows.append((tf.date(), k, round(cand[k], 2), round(rel, 1), sess_date.date(), bool(flag.get(sess_date, False)),
                     [d.date() for d in flag[(flag.index >= lo) & (flag.index <= hi + pd.Timedelta(days=2))].index[flag[(flag.index >= lo) & (flag.index <= hi + pd.Timedelta(days=2))]]]))
R = pd.DataFrame(rows, columns=['expiry', 'splice_ts', 'jump_pts', 'jump_bp', 'splice_session', 'flagged?', 'project_flagged_in_window'])
pd.set_option('display.width', 250)
print(R.to_string())
print('splice session matches project flag:', int(R['flagged?'].sum()), 'of', len(R))
R.to_pickle(f'/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/true_splice_{mkt}.pkl')
