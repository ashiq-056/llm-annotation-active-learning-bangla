"""Figures for the SentiGOLD LLM-annotator experiment.

Usage: python analysis/plot_3b.py --results results/sentigold_3b --out figures
"""
import argparse
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/sentigold_3b")
ap.add_argument("--out", default="figures")
_a = ap.parse_args()
D, OUT = _a.results, _a.out
import os; os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"random": "#2a78d6", "margin": "#eb6834", "entropy": "#1baf7a"}     # slots 1-3, validated all-pairs
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF})

def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)

curve = {(a, s): pd.read_csv(f"{D}/curves/sentigold_{a}_annotator_{s}_curve.csv") for a in ("human", "llm") for s in COL}
ann = {s: pd.read_csv(f"{D}/annotations_anon/sentigold_llm_annotator_{s}_annotations_anon.csv") for s in COL}

# ---------------- Figure 1: learning curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
for ax, (col, title) in zip(axes, [("test_f1_macro", "Test macro-F1"), ("test_accuracy", "Test accuracy")]):
    for (a, s), c in curve.items():
        ax.plot(c.train_size, c[col], color=COL[s], lw=2, ls="-" if a == "human" else (0, (5, 3)), solid_capstyle="round")
    style(ax); ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("Labeled training examples (200 seed + 100 per round)", fontsize=10)
    ax.set_xticks(range(200, 1601, 200)); ax.set_ylim(0.30 if col == "test_f1_macro" else 0.40, 0.80)
sty = [Line2D([0], [0], color=COL[s], lw=2, label=s) for s in COL]
ann_l = [Line2D([0], [0], color=INK2, lw=2, ls="-", label="human labels (gold)"),
         Line2D([0], [0], color=INK2, lw=2, ls=(0, (5, 3)), label="LLM labels (Qwen2.5-7B, zero-shot)")]
leg1 = fig.legend(handles=sty, title="Query strategy", loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.06, -0.02), fontsize=10, title_fontsize=10)
fig.legend(handles=ann_l, title="Who labels the queried rows", loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(0.97, -0.02), fontsize=10, title_fontsize=10)
fig.suptitle("SentiGOLD, BanglaBERT active learning: human vs LLM annotator (single seed, 1,570 test rows)", x=0.06, ha="left", fontsize=12.5, y=1.0)
fig.tight_layout(rect=(0, 0.08, 1, 0.97)); fig.savefig(f"{OUT}/fig1_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: why the LLM curves flatten ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
for s, c in COL.items():
    a = ann[s]; ag = (a.true_label == a.train_label).groupby(a["round"]).mean()
    a1.plot(ag.index * 100 + 200, ag.values, color=c, lw=2)
    ytxt = {"random": ag.values[-1], "margin": 0.535, "entropy": 0.435}[s]      # nudged apart so labels never collide
    a1.text(1600 + 25, ytxt, f"{s}: {(a.true_label == a.train_label).mean():.0%} overall", color=INK2, fontsize=9.5, va="center")
style(a1); a1.set_ylim(0.3, 0.9); a1.set_xlim(180, 2000); a1.set_xticks(range(200, 1501, 200))
a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a1.set_title("LLM label agreement with gold, rows queried in each round (100 rows/round)", loc="left", fontsize=11, pad=10)
a1.set_xlabel("Labeled examples before the round's query", fontsize=10)

xs = np.arange(3)
for i, (s, c) in enumerate(COL.items()):
    a = ann[s]; gold = (a.true_label == 0).mean(); llm = (a.train_label == 0).mean()
    a2.plot([i, i], [gold, llm], color=c, lw=2)
    a2.plot(i, gold, "o", ms=10, mfc=SURF, mec=c, mew=2)
    a2.plot(i, llm, "o", ms=10, mfc=c, mec=c)
style(a2); a2.set_xticks(xs); a2.set_xticklabels(list(COL)); a2.set_xlim(-0.6, 2.6); a2.set_ylim(0, 0.55)
a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a2.set_title("Share of queried rows that are 'neutral'", loc="left", fontsize=11, pad=10)
a2.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=10, mfc=SURF, mec=INK2, mew=2, label="gold label"),
                   Line2D([0], [0], marker="o", ls="", ms=10, mfc=INK2, mec=INK2, label="LLM label")],
          loc="upper left", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_annotator.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
