"""Weekly PR18 verdict straight from the live journal (the state pickle) and results/forward_thresholds_R.json: no Databento and no bars file needed.
   python -m paper.verdict --state paper_state.pkl [--thresholds results/forward_thresholds_R.json]
Same rules as paper/forward_test.py (kill_reasons): looks every 5 sessions from 60 up to 250 (the stored thresholds end at 250), latched; flagged sessions stay in the sums (PR18) and are listed."""
from __future__ import annotations

import argparse
import json
import math
import pickle

MU, SD = 0.0183, 0.232            # development pooled R per session and its block-bootstrap sd (council 2 ops, computed from the re-based development R)


def weekly_verdict(journal, thresholds) -> dict:
    ok = lambda r: math.isfinite(r.get("atr_usd", float("nan"))) and r["atr_usd"] > 0
    Rn = [r["noise"]["usd"] / r["atr_usd"] if ok(r) else 0.0 for r in journal]
    Rg = [r["gap"]["usd"] / r["atr_usd"] if ok(r) else 0.0 for r in journal]
    P = [(3 * a + 2 * b) / 5 for a, b in zip(Rn, Rg)]
    L = len(P)
    first, reasons = None, []
    for k in range(60, min(L, 250) + 1, 5):
        s = sum(P[:k])
        if s < thresholds[str(k)]:
            first, reasons = k, [f"pooled R {s:+.2f} < dev 5th percentile {thresholds[str(k)]:+.2f} at session {k}"]
            break
        if k >= 120 and sum(Rn[:k]) < 0 and sum(Rg[:k]) < 0:
            first, reasons = k, [f"both sleeves negative at session {k}"]
            break
    holes = [r["date"] for r in journal if r.get("data_gap")]
    S = sum(P)
    return dict(sessions=L, pooled_R=S, noise_R=sum(Rn), gap_R=sum(Rg), first_stop_session=first, reasons=reasons, sessions_with_holes=len(holes), hole_share=len(holes) / max(L, 1), holes=holes,
                bayes_factor_full_vs_zero=math.exp((MU / SD ** 2) * (S - MU * L / 2)), beyond_reference=L > 250,
                verdict=("INVALID" if holes and len(holes) / L > 0.02 else "STOP" if first else "CONTINUE" if L >= 60 else "TOO EARLY"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="paper_state.pkl")
    ap.add_argument("--thresholds", default="results/forward_thresholds_R.json")
    a = ap.parse_args(argv)
    th = json.load(open(a.thresholds))
    j = pickle.load(open(a.state, "rb"))["journal"]
    print(json.dumps(weekly_verdict(j, th), indent=1))


if __name__ == "__main__":
    main()
