"""PR19 forward-test verdict straight from the live journal (the harness state file); no Databento and no bars file needed.
   python -m paper.verdict --state paper_state.pkl [--thresholds results/forward_thresholds_R.json]
Rules are those of docs/PREREGISTRATION_19.md and nothing here may be tuned during the test:
  R = session P&L per micro / ATR in $ per micro; pooled R per session = (3 R_noise + 2 R_gap) / 5; t = sessions counted since day 0 (day 0 = first journal row).
  Looks: every 21 sessions from session 63 (month 3) to 756 (month 36).
  KILL     at a valid look k: cumulative pooled R <= -5.75 + 0.009 k (latched). It means stop and investigate; it is not proof that the edge is dead.
  CONFIRM  only at session 756, valid, never killed, cumulative pooled R >= 10.89 (z >= 1.645 against zero edge at the development sd of 0.2407 per session). Anything else at 756 is 'cannot tell'.
  WATCH    informational: pooled R below the PR18 development 5th percentile (weekly, sessions 60 to 250), or both sleeves negative after session 120. Never a stop.
  Data     a session whose regular hours have holes (journal field data_gap; exchange early closes excepted) counts as a hole session. Gate at session 20: two or more hole sessions among the
           first 20 ends the test as DATA FAILURE (a feed problem, not a verdict on the strategy). From session 63 on a look is INVALID (no kill evaluated) when more than 2% of the counted
           sessions are hole sessions; two INVALID looks in a row end the test as DATA FAILURE. Flagged sessions stay in the sums.
  Shadow columns (descriptive, never a decision): R of the nq_vwap sleeve and of the frozen equal-weight basket (R_noise + R_vwap + R_gap) / 3.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from typing import Dict, List, Optional

H19, SLOPE19 = 5.75, 0.009
FIRST_LOOK, EVERY, HORIZON = 63, 21, 756
SIGMA_DEV, MU_DEV = 0.2407, 0.0183                  # development pooled R per session: sd and mean (re-based, council 2 statistician)
CONFIRM_R = 1.645 * SIGMA_DEV * math.sqrt(HORIZON)  # 10.89
HOLE_SHARE_MAX, GATE_SESSION, GATE_HOLES = 0.02, 20, 2


def kill_threshold(k: int) -> float:
    return -H19 + SLOPE19 * k


def _r(row: dict, key: str) -> float:
    atr = row.get("atr_usd", float("nan"))
    v = row.get(key)
    if v is None or not (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0):
        return 0.0
    return v["usd"] / atr


def verdict19(journal: List[dict], thresholds18: Optional[Dict[str, float]] = None) -> dict:
    Rn, Rg = [_r(r, "noise") for r in journal], [_r(r, "gap") for r in journal]
    Rv = [_r(r, "vwap") for r in journal]
    P = [(3 * a + 2 * b) / 5 for a, b in zip(Rn, Rg)]
    basket = [(a + b + c) / 3 for a, b, c in zip(Rn, Rv, Rg)]
    hole = [bool(r.get("data_gap")) for r in journal]
    L = len(P)
    cum = lambda x, k: sum(x[:k])
    status, reason, kill_at, data_failure_at = None, "", None, None
    invalid_looks: List[int] = []
    if L >= GATE_SESSION and sum(hole[:GATE_SESSION]) >= GATE_HOLES:
        status, reason, data_failure_at = "DATA FAILURE", f"{sum(hole[:GATE_SESSION])} hole sessions among the first {GATE_SESSION} (gate: fewer than {GATE_HOLES})", GATE_SESSION
    prev_invalid = False
    if status is None:
        for k in range(FIRST_LOOK, min(L, HORIZON) + 1, EVERY):
            share = sum(hole[:k]) / k
            if share > HOLE_SHARE_MAX:
                invalid_looks.append(k)
                if prev_invalid:
                    status, reason, data_failure_at = "DATA FAILURE", f"two invalid looks in a row (session {k}: {share:.1%} hole sessions)", k
                    break
                prev_invalid = True
                continue
            prev_invalid = False
            if cum(P, k) <= kill_threshold(k):
                status, reason, kill_at = "KILL", f"cumulative pooled R {cum(P, k):+.2f} <= {kill_threshold(k):+.2f} at session {k}", k
                break
    S = cum(P, L)
    last_look = max([k for k in range(FIRST_LOOK, min(L, HORIZON) + 1, EVERY)], default=None)
    watch = []
    if thresholds18:
        for k in range(60, min(L, 250) + 1, 5):
            if str(k) in thresholds18 and cum(P, k) < thresholds18[str(k)]:
                watch.append(f"pooled R {cum(P, k):+.2f} below the PR18 development 5th percentile {thresholds18[str(k)]:+.2f} at session {k}")
                break
    if L >= 120 and cum(Rn, L) < 0 and cum(Rg, L) < 0:
        watch.append("both sleeves have negative cumulative R")
    if status is None:
        if L < FIRST_LOOK:
            status = "TOO EARLY"
        elif L >= HORIZON:
            if last_look is not None and last_look >= HORIZON - EVERY + 1 and sum(hole[:HORIZON]) / HORIZON <= HOLE_SHARE_MAX:
                status = "CONFIRMED (36 months)" if cum(P, HORIZON) >= CONFIRM_R else "CANNOT TELL (36 months)"
                reason = f"cumulative pooled R {cum(P, HORIZON):+.2f} against the confirm line {CONFIRM_R:.2f}"
            else:
                status = "INVALID"
        elif invalid_looks and invalid_looks[-1] == last_look:
            status = "INVALID"
        else:
            status = "WATCH" if watch else "CONTINUE"
    nxt = None if L >= HORIZON else (FIRST_LOOK if L < FIRST_LOOK else FIRST_LOOK + EVERY * ((L - FIRST_LOOK) // EVERY + 1))
    return dict(status=status, reason=reason, sessions=L, pooled_R=S, noise_R=cum(Rn, L), gap_R=cum(Rg, L), shadow_vwap_R=cum(Rv, L), shadow_basket_R=cum(basket, L),
                kill_threshold_now=kill_threshold(L) if L else None, next_look=nxt, kill_session=kill_at, data_failure_session=data_failure_at, invalid_looks=invalid_looks,
                hole_sessions=sum(hole), hole_share=sum(hole) / L if L else 0.0, watch=watch, confirm_line=CONFIRM_R,
                likelihood_ratio_full_vs_zero=math.exp((MU_DEV / SIGMA_DEV ** 2) * (S - MU_DEV * L / 2)) if L else 1.0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="paper_state.pkl")
    ap.add_argument("--thresholds", default="results/forward_thresholds_R.json")
    a = ap.parse_args(argv)
    try:
        th = json.load(open(a.thresholds))
    except OSError:
        th = None
    j = pickle.load(open(a.state, "rb"))["journal"]
    print(json.dumps(verdict19(j, th), indent=1, default=float))


if __name__ == "__main__":
    main()
