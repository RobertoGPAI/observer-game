"""Paper 1 figures, recomputed from the raw corpus -- like analysis.py, the
figures come from one committed script or they are not figures.

Reads results/observer_game.jsonl, writes figures/fig{1,2,3}.{pdf,png}.
PDF is the LaTeX-included vector; PNG is for quick review.

Palette: Okabe-Ito blue (#0072B2, governed) / vermillion (#D55E00,
ungoverned) -- CVD-validated pair (worst adjacent dE 91.9, protan).
Identity is never color-alone: panels and direct labels carry it too.

Usage:  python paper_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single source of truth: the committed parser and the committed fit.
# Mirroring them here is how a figure ends up contradicting a table.
from analysis import MODEL_26B, self_estimate

BLUE = "#0072B2"    # governed / landed
VERM = "#D55E00"    # ungoverned / decapitated
INK = "#333333"
MUTED = "#888888"

BUDGET = 12_000
OUT = Path("figures")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load() -> list[dict]:
    with open("results/observer_game.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def is_wall(cell: str) -> bool:
    return cell in ("wall+meter", "wall-silent")


# ---------------------------------------------------------------------------
# Fig 1 -- Only the wall binds: every run's spend, by cell and family.
# ---------------------------------------------------------------------------

# Unwalled cells ordered by Gemma median (descending temptation), walls last.
CELL_ORDER = ["oracle", "cheap-talk", "free", "panopticon", "bluff",
              "blind-covenant", "covenant", "wall+meter", "wall-silent"]


def fig1(rows: list[dict]) -> None:
    fams = [("Gemma 4 26B", [r for r in rows if r["model"] == MODEL_26B]),
            ("Nemotron 3 Ultra 550B", [r for r in rows if "nemotron" in r["model"]])]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    dropped = 0
    for ax, (name, fam) in zip(axes, fams):
        for i, cell in enumerate(CELL_ORDER):
            # zero-spend runs are provider casualties, not spending data:
            # clamping them onto a log axis fabricates a "1-token mission".
            runs = [r for r in fam if r["cell"] == cell and r["spent_task"] > 0]
            dropped += sum(1 for r in fam
                           if r["cell"] == cell and r["spent_task"] <= 0)
            if not runs:
                continue
            color = BLUE if is_wall(cell) else VERM
            # deterministic horizontal jitter: seed on run index
            xs = [i + (((hash((cell, j)) % 1000) / 1000 - 0.5) * 0.55)
                  for j in range(len(runs))]
            ax.scatter(xs, [r["spent_task"] for r in runs],
                       s=9, facecolors="none" if is_wall(cell) else color,
                       edgecolors=color, linewidths=0.7, alpha=0.85, zorder=3)
        ax.axhline(BUDGET, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(range(len(CELL_ORDER)))
        ax.set_xticklabels(CELL_ORDER, rotation=38, ha="right", fontsize=7)
        ax.set_title(name, loc="left")
        ax.grid(axis="y", color="#DDDDDD", lw=0.4, zorder=0)
    axes[0].set_ylabel("mission spend (tokens, log)")
    axes[1].text(len(CELL_ORDER) - 0.4, BUDGET * 1.25, "budget 12k",
                 fontsize=7, color=INK, ha="right")
    axes[0].text(2.0, 160_000, "unwalled", fontsize=7.5, color=VERM,
                 ha="center")
    axes[0].text(7.5, 1_700, "walled", fontsize=7.5, color=BLUE, ha="center")
    print(f"fig1: {dropped} zero-spend provider casualties excluded")
    fig.savefig(OUT / "fig1_wall_binds.pdf")
    fig.savefig(OUT / "fig1_wall_binds.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 -- Decapitation and landing: deliverable size in the 84 walled runs.
# ---------------------------------------------------------------------------

def fig2(rows: list[dict]) -> None:
    walls = [r for r in rows if r["enforce"] and is_wall(r["cell"])]
    pre = [r["report_chars"] for r in walls if r["report_chars"] <= 420]
    post = [r["report_chars"] for r in walls if r["report_chars"] > 420]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    bins = [10 ** (e / 12) for e in range(24, 60)]  # log bins 100..~30k
    ax.hist(pre, bins=bins, color=VERM, edgecolor="white", lw=0.5,
            label=f"terminal denial (n={len(pre)})")
    ax.hist(post, bins=bins, color=BLUE, edgecolor="white", lw=0.5,
            label=f"landing protocol (n={len(post)})")
    ax.set_xscale("log")
    ax.set_xlabel("final deliverable (characters, log)")
    ax.set_ylabel("missions")
    ax.annotate("402 chars =\nthe denial text itself", xy=(402, 22),
                xytext=(1_100, 20), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", color="#DDDDDD", lw=0.4, zorder=0)
    ax.set_title("Same wall, zero overshoot on both sides", loc="left")
    fig.savefig(OUT / "fig2_decapitation.pdf")
    fig.savefig(OUT / "fig2_decapitation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 -- Empty introspection: self-estimate vs ledger truth (log-log).
# ---------------------------------------------------------------------------

def fig3(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    series = [
        ("Gemma 4 26B", lambda r: r["model"] == MODEL_26B,
         dict(marker="o", facecolors=INK, edgecolors="none", s=10, alpha=0.55)),
        ("Nemotron 3 Ultra", lambda r: "ultra" in r["model"],
         dict(marker="^", facecolors="none", edgecolors=BLUE, s=16,
              linewidths=0.8, alpha=0.85)),
    ]
    pairs = []  # global fit set = the canonical one (sin 31b, spent > 100)
    for label, keep, style in series:
        pts = [(r["spent_task"], est) for r in rows
               if keep(r) and r["spent_task"] > 100
               and (est := self_estimate(r))]
        pairs += pts
        ax.scatter([x for x, _ in pts], [y for _, y in pts],
                   label=f"{label} (n={len(pts)})", zorder=3, **style)
    lo, hi = 100, 1_000_000
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(200_000, 320_000, "y = x", fontsize=7, color=MUTED, rotation=38)
    ax.axhline(BUDGET, color=VERM, lw=0.7, ls=":", zorder=2)
    ax.text(130, BUDGET * 1.35, "the anchor: budget 12k", fontsize=7,
            color=VERM)
    # The fit is computed HERE from the same pairs (natural log, as in
    # analysis.py) -- a hardcoded line once drifted a full decade off the
    # cloud because it assumed the wrong log base.
    n = len(pairs)
    xs = [math.log(x) for x, _ in pairs]
    ys = [math.log(y) for _, y in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ssr / sst
    grid = [lo, hi]
    ax.plot(grid, [math.exp(a) * x ** b for x in grid],
            color=INK, lw=1.1, zorder=2)
    ax.text(0.97, 0.97, f"fit: slope {b:.2f}, $r^2$={r2:.2f} (n={n})",
            transform=ax.transAxes, fontsize=7, color=INK,
            ha="right", va="top")
    print(f"fig3 fit: log(est) = {a:.2f} + {b:.2f}*log(real), "
          f"r2={r2:.2f}, n={n}  <- must match analysis.py")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual spend (tokens, log)")
    ax.set_ylabel("self-estimate (tokens, log)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(color="#DDDDDD", lw=0.4, zorder=0)
    fig.savefig(OUT / "fig3_introspection.pdf")
    fig.savefig(OUT / "fig3_introspection.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load()
    fig1(rows)
    fig2(rows)
    fig3(rows)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig*.p*"))))


if __name__ == "__main__":
    main()
