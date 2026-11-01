"""Summary tables for all SentiGOLD active-learning runs (curves + anonymized annotation logs).

Usage: python analysis/summarize_runs.py --results results
Needs no GPU and no dataset text.
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
D = ap.parse_args().results
SEEDS = (17, 18)
KINDS = ("random", "margin", "entropy", "balanced_random")
RUNS = [(a, s) for a in ("human", "llm", "llm_fs") for s in KINDS if not (a == "llm" and s == "balanced_random")]
pd.set_option("display.width", 200)

rec = []
for seed in SEEDS:
    for a, s in RUNS:
        rid = f"sentigold_s{seed}_{a}_{s}"
        cp, ap_ = f"{D}/curves/{rid}_curve.csv", f"{D}/annotations_anon/{rid}_annotations_anon.csv"
        if not (os.path.exists(cp) and os.path.exists(ap_)):
            print("missing:", rid); continue
        c, an = pd.read_csv(cp), pd.read_csv(ap_)
        if len(c) < 15:
            print(f"unfinished ({len(c)} rounds): {rid}")
        late = c[c["round"] >= 10]
        hours = c["round_seconds"].sum() / 3600 if "round_seconds" in c else np.nan
        rec.append(dict(seed=seed, annotator=a, strategy=s, f1=late.test_f1_macro.mean(), acc=late.test_accuracy.mean(),
                        agree=(an.true_label == an.train_label).mean(), neu_label=(an.train_label == 0).mean(),
                        neu_gold=(an.true_label == 0).mean(), hours=hours))
R = pd.DataFrame(rec)
order = pd.MultiIndex.from_tuples(RUNS, names=["annotator", "strategy"])

def pivot(col):
    return R.pivot_table(index=["annotator", "strategy"], columns="seed", values=col).reindex(order).dropna(how="all")

def table(col, title, only=None):
    P = pivot(col)
    if only:
        P = P[P.index.get_level_values(0).isin(only)]
    per_seed = P.copy()
    P["mean"] = per_seed.mean(axis=1); P["spread"] = per_seed.max(axis=1) - per_seed.min(axis=1)
    print("\n" + title); print(P.apply(lambda x: x.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")).to_string())

def line(vals):
    return "  ".join(f"{v:+.1f}" for v in vals) + "   | mean " + f"{np.mean(vals):+.1f}"

table("f1", "macro-F1, mean of rounds 10-14 (spread = max - min over seeds)")
table("acc", "accuracy, mean of rounds 10-14")
table("agree", "label agreement with gold on queried rows", only=("llm", "llm_fs"))
table("neu_label", "share of labels that are NEUTRAL", only=("llm", "llm_fs"))
table("neu_gold", "gold neutral share of the queried rows", only=("llm_fs",))

print("\nGAINS in F1 points per seed, then mean (same annotator):")
for a in ("human", "llm", "llm_fs"):
    for base in ("random", "balanced_random"):
        if a == "llm" and base == "balanced_random":
            continue
        for s in ("margin", "entropy"):
            vals = []
            for seed in SEEDS:
                x = R[(R.seed == seed) & (R.annotator == a)].set_index("strategy")["f1"]
                if s in x and base in x:
                    vals.append(100 * (x[s] - x[base]))
            if vals:
                print(f"  {a:7s} {s:8s} vs {base:15s}: " + line(vals))

print("\nGAP to human labels in F1 points (human minus annotator):")
for a in ("llm", "llm_fs"):
    for s in ("random", "margin", "entropy"):
        vals = []
        for seed in SEEDS:
            r_ = R[(R.seed == seed) & (R.strategy == s)]
            h, x = r_[r_.annotator == "human"]["f1"], r_[r_.annotator == a]["f1"]
            if len(h) and len(x):
                vals.append(100 * (h.iloc[0] - x.iloc[0]))
        if vals:
            print(f"  {a:7s} {s:8s}: " + line(vals))

H = pivot("hours")
print("\nhours per run (only runs with timing recorded):"); print(H.round(2).to_string())
print(f"total recorded: {np.nansum(H.values):.1f} h")
