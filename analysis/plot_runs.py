"""Figures for the SentiGOLD active-learning runs.

Usage: python analysis/plot_runs.py --results results --out figures
  fig1_curves.png    macro-F1 vs labels, one panel per annotator, mean over seeds 17 and 18 (band = min..max of the two seeds)
  fig2_summary.png   macro-F1 (mean of rounds 10-14) for every run; dots = seeds, bar = mean
  fig3_annotator.png label agreement with gold and neutral share of the annotator's labels
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
a_ = ap.parse_args()
D, OUT = a_.results, a_.out
os.makedirs(OUT, exist_ok=True)
SEEDS = (17, 18)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"random": "#2a78d6", "margin": "#eb6834", "entropy": "#1baf7a", "balanced_random": "#7a5bc7"}
LABEL = {"random": "random", "margin": "margin", "entropy": "entropy", "balanced_random": "class-balanced random"}
ANN = {"human": "human labels (gold)", "llm": "LLM zero-shot", "llm_fs": "LLM few-shot + calibration"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF})

def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

def kinds(a):
    return [k for k in COL if not (a == "llm" and k == "balanced_random")]

def curve(a, k, seed):
    return pd.read_csv(f"{D}/curves/sentigold_s{seed}_{a}_{k}_curve.csv")

def ann(a, k, seed):
    return pd.read_csv(f"{D}/annotations_anon/sentigold_s{seed}_{a}_{k}_annotations_anon.csv")

# ---- Figure 1: learning curves ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, a in zip(axes, ANN):
    for k in kinds(a):
        M = np.vstack([curve(a, k, s).test_f1_macro.values for s in SEEDS])
        x = curve(a, k, SEEDS[0]).train_size.values
        ax.fill_between(x, M.min(0), M.max(0), color=COL[k], alpha=0.12, lw=0)
        ax.plot(x, M.mean(0), color=COL[k], lw=2, label=LABEL[k])
    style(ax); ax.set_title(ANN[a], loc="left", fontsize=11.5, pad=10)
    ax.set_xticks(range(200, 1601, 400)); ax.set_ylim(0.30, 0.78)
    ax.set_xlabel("Labeled examples (200 seed + 100 per round)", fontsize=9.5)
axes[0].set_ylabel("Test macro-F1", fontsize=10)
axes[2].legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("SentiGOLD, BanglaBERT active learning (line = mean of seeds 17 and 18, band = range of the two seeds)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig1_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: summary of every run ----
fig, ax = plt.subplots(figsize=(11, 4.8))
x0, ticks, tl = 0, [], []
for a in ANN:
    ks = kinds(a)
    for i, k in enumerate(ks):
        v = [curve(a, k, s).query("round>=10").test_f1_macro.mean() for s in SEEDS]
        ax.bar(x0 + i, np.mean(v), color=COL[k], alpha=0.55, width=0.7)
        ax.scatter([x0 + i] * 2, v, color=COL[k], edgecolor=INK, lw=0.8, zorder=3, s=28)
    ticks.append(x0 + (len(ks) - 1) / 2); tl.append(ANN[a]); x0 += len(ks) + 1
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=10); style(ax); ax.set_ylim(0.55, 0.78)
ax.set_ylabel("Test macro-F1, mean of rounds 10-14", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL[k], alpha=0.7, label=LABEL[k]) for k in COL], frameon=False, ncol=4, loc="upper right", fontsize=9)
ax.set_title("Bars = mean of two seeds, dots = each seed", loc="left", fontsize=11, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_summary.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: annotator behaviour ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
w, xs = 0.38, np.arange(3)
for j, (a, hatch) in enumerate((("llm", ""), ("llm_fs", "//"))):
    for i, k in enumerate(("random", "margin", "entropy")):
        ag = np.mean([(lambda d: (d.true_label == d.train_label).mean())(ann(a, k, s)) for s in SEEDS])
        a1.bar(i + (j - 0.5) * w, ag, w, color=COL[k], alpha=0.8, hatch=hatch, edgecolor=SURF)
        a1.text(i + (j - 0.5) * w, ag + 0.01, f"{ag:.0%}", ha="center", fontsize=9, color=INK2)
a1.set_xticks(xs); a1.set_xticklabels(["random", "margin", "entropy"]); a1.set_ylim(0, 0.9); style(a1)
a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a1.set_title("Label agreement with gold on queried rows\n(left bar zero-shot, hatched bar few-shot + calibration)", loc="left", fontsize=10.5, pad=10)
for j, a in enumerate(("llm", "llm_fs")):
    for i, k in enumerate(("random", "margin", "entropy")):
        d = pd.concat([ann(a, k, s) for s in SEEDS])
        gold, lab = (d.true_label == 0).mean(), (d.train_label == 0).mean()
        xx = i + (j - 0.5) * 0.3
        a2.plot([xx, xx], [gold, lab], color=COL[k], lw=2)
        a2.plot(xx, gold, "o", ms=9, mfc=SURF, mec=COL[k], mew=2)
        a2.plot(xx, lab, "o" if a == "llm" else "s", ms=9, mfc=COL[k], mec=COL[k])
a2.set_xticks(xs); a2.set_xticklabels(["random", "margin", "entropy"]); a2.set_ylim(0, 0.6); style(a2)
a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a2.set_title("Share of queried rows that are neutral\n(open = gold, filled circle = zero-shot, filled square = few-shot + calibration)", loc="left", fontsize=10.5, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_annotator.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
