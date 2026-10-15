"""Find the instrument in the London Strategic Edge vault.   LSE_API_KEY=... python -m paper.lse_probe NQ   (prints catalog rows whose symbol or name contains the pattern and the span of their history).
The vault's 'Futures' class is labelled 'eurex' in the SDK; if CME NQ / MNQ is not listed, the harness cannot trade this feed (an index or ETF proxy has no 18:00-16:00 session and different volume)."""
import sys

from paper.feed import lse_catalog, lse_find

if __name__ == "__main__":
    pats = sys.argv[1:] or ["NQ"]
    cat = lse_catalog()
    print(f"{len(cat)} instruments in the vault; datasets: {sorted({r.get('dataset') for r in cat})}")
    for p in pats:
        rows = lse_find(p, catalog=cat)
        print(f"\n'{p}': {len(rows)} matches")
        for r in rows[:40]:
            print("  ", {k: r.get(k) for k in ("dataset", "symbol", "name", "first_tick", "last_tick", "ticks")})
