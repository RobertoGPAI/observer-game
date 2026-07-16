"""Reproducible analysis for the observer-game corpus: the paper's numbers.

Every headline figure in the docs must come out of THIS script run against
results/observer_game.jsonl -- if it doesn't, the doc is wrong, not the data.
Born of an audit (2026-07-15) that caught two bugs in ad-hoc session
analysis: a quick median that took the UPPER of the central pair on even n
(the exact bug documented and fixed in InputCalibrator.factor -- the analyst
caught the governor's disease), and a family grouping ('gemma' in model)
that silently mixed the 24 gemma-4-31b sibling runs into the 26B tables.

Usage:  python analysis.py [results/observer_game.jsonl]
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict

MODEL_26B = "gemma-4-26b-a4b-it"

CELL_ORDER = ["free", "cheap-talk", "panopticon", "bluff", "covenant",
              "blind-covenant", "oracle", "wall+meter", "wall-silent",
              "custom"]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def overshoot_census(rows: list[dict]) -> None:
    """The wall claim, stated exactly: named cells vs the custom incident."""
    enforced = [r for r in rows if r["enforce"]]
    named = [r for r in enforced if r["cell"] != "custom"]
    over = [r for r in enforced if r["overshoot"] > 0]
    print(f"enforce runs: {len(enforced)} "
          f"(named cells {len(named)}, custom {len(enforced) - len(named)})")
    print(f"named-cell overshoots: "
          f"{sum(1 for r in named if r['overshoot'] > 0)}/{len(named)}")
    for r in over:
        print(f"  overshoot: {r['overshoot']} tokens, cell={r['cell']}, "
              f"model={r['model']}, at={r['at']}"
              "  <- the governor's own chars//4 miscalibration incident")


def per_cell_table(rows: list[dict], model: str = MODEL_26B) -> None:
    """Median spend and overshoot rate per cell, ONE model, proper median."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            cells[r["cell"]].append(r)
    print(f"{'cell':<16} {'n':>3} {'median spent':>12} {'overshoot':>10}")
    for cell in CELL_ORDER:
        rs = cells.get(cell, [])
        if not rs:
            continue
        med = statistics.median(r["spent_task"] for r in rs)
        ov = sum(1 for r in rs if r["overshoot"] > 0)
        print(f"{cell:<16} {len(rs):>3} {med:>12.0f} {ov:>7}/{len(rs)}")


def self_estimate(r: dict) -> int | None:
    """The documented parser: LAST token-figure in the audit reply.

    Reasoning models deliberate first and conclude last, so the final
    number adjacent to the word 'token(s)' is the claim; figures equal to
    the budget are the anchor restated, not an estimate, and are skipped.
    A naive first-number parser yields slope 0.24 / r2 0.01 on the same
    corpus -- parser choice moves coefficients, never the conclusion
    (weak coupling, mass underestimation).
    """
    text = r.get("audit_reply") or ""
    hits = [(m.start(), int(m.group(1).replace(",", "")))
            for m in re.finditer(r"([\d,]{3,})\s*(?:total\s+)?tokens?",
                                 text, re.I)]
    hits = [(pos, n) for pos, n in hits
            if 100 <= n <= 2_000_000 and n != r["budget"]]
    return hits[-1][1] if hits else None


def regression(rows: list[dict]) -> None:
    """log(self-estimate) on log(actual spend), excluding the 31b sibling."""
    def fit(pairs):
        n = len(pairs)
        xs = [math.log(x) for x, _ in pairs]
        ys = [math.log(y) for _, y in pairs]
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx
        a = my - b * mx
        ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sst = sum((y - my) ** 2 for y in ys)
        return a, b, 1 - ssr / sst, n

    families = (
        ("gemma-26b", lambda r: r["model"] == MODEL_26B),
        ("ultra", lambda r: "ultra" in r["model"]),
        ("global (sin 31b)", lambda r: "31b" not in r["model"]),
    )
    for label, keep in families:
        pairs = [(r["spent_task"], est) for r in rows
                 if keep(r) and r["spent_task"] > 100
                 and (est := self_estimate(r))]
        if len(pairs) < 4:
            continue
        a, b, r2, n = fit(pairs)
        under = sum(1 for x, y in pairs if y < x)
        print(f"[{label:<17}] log(est) = {a:.2f} + {b:.2f}*log(real)  "
              f"r2={r2:.2f}  n={n}  underestimate={under}/{n} "
              f"({100 * under // n}%)")


def thinking_halves(rows: list[dict]) -> None:
    """The Ultra on/off halves ARE separable: the `preamble` field."""
    ultra = [r for r in rows if "ultra" in r["model"]]
    by = defaultdict(list)
    for r in ultra:
        by[r.get("preamble", "(sin campo)")].append(r)
    for pre, rs in sorted(by.items()):
        print(f"  preamble={pre!r}: {len(rs)} runs")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/observer_game.jsonl"
    rows = load(path)
    print(f"corpus: {len(rows)} runs\n")
    print("== 1. Overshoot census (the wall claim) ==")
    overshoot_census(rows)
    print("\n== 2. Per-cell table (gemma-4-26b only, statistics.median) ==")
    per_cell_table(rows)
    print("\n== 3. Self-estimate regression (documented parser) ==")
    regression(rows)
    print("\n== 4. Ultra thinking halves (separable via `preamble`) ==")
    thinking_halves(rows)


if __name__ == "__main__":
    main()
