"""Audit: every trade of every sleeve in use must be opened and closed inside US regular trading hours (09:30 ET = smin 930 to 15:55 ET = smin 1315).
Nothing is held overnight and nothing is traded in the Globex/overnight session. Signals may USE overnight prices (gap, overnight range, 18:00-anchored VWAP) but no position exists then."""
import sys
import numpy as np
import intraday_strategies as st
from portfolio import make_sleeve
from run_corrected import NEW

C = st.CASH_OPEN
SLEEVES = [("nq", "noise_area", {}, {}), ("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=C), {}), ("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2), {}),
           ("es", "noise_area", {}, {}), ("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=C), dict(trend=250)), ("gc", "noise_area", {}, dict(trend=60)),
           ("es", "xs_prevday", dict(buf=0.0, s=0.3), {}), ("es", "xs_onrange", dict(buf=0.1, s=0.4, anchor=C), {}), ("gc", "xs_orb", dict(or_min=60, s=1.0, anchor=860), {}), ("gc", "gap", dict(thr=0.5, s=0.5, mode=-1), {})]
bad = 0
for m, sl, params, gate in SLEEVES:
    st.TRACE = []
    make_sleeve(m, sl, params, gate, costs=NEW, name="audit")
    tr = np.array(st.TRACE); st.TRACE = None
    if sl == "vv_filt":     # vv_filt calls vwap_atr_session, which logs
        pass
    lo, hi = (tr[:, 0].min(), tr[:, 1].max()) if len(tr) else (None, None)
    ok = len(tr) > 0 and tr[:, 0].min() >= C and tr[:, 1].max() <= 1315 and (tr[:, 1] > tr[:, 0]).all()
    flag = "OK " if ok else "BAD"
    if not ok: bad += 1
    print(f"{flag} {m} {sl:<11} {str(params)[:46]:<46} trades {len(tr):6d} | earliest entry {lo} ({(lo//60+18)%24:02d}:{lo%60:02d} ET) | latest exit {hi} ({(hi//60+18)%24:02d}:{hi%60:02d} ET)" if len(tr) else f"{flag} {m} {sl} no trades")
print("\nALL SLEEVES TRADE ONLY BETWEEN 09:30 AND 15:55 ET" if bad == 0 else f"\n{bad} SLEEVE(S) TRADE OUTSIDE REGULAR HOURS")
