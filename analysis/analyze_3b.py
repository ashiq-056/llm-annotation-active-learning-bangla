"""Summarize the SentiGOLD LLM-annotator experiment (curves + anonymized annotation logs).

Usage:  python analysis/analyze_3b.py --results results/sentigold_3b
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: F401 (kept for extensions)

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/sentigold_3b")
D = p.parse_args().results
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

runs = [(a, s) for a in ("human", "llm") for s in ("random", "margin", "entropy")]
curve = {r: pd.read_csv(f"{D}/curves/sentigold_{r[0]}_annotator_{r[1]}_curve.csv") for r in runs}
ann = {r: pd.read_csv(f"{D}/annotations_anon/sentigold_{r[0]}_annotator_{r[1]}_annotations_anon.csv") for r in runs}
names = {0: "neutral", 1: "positive", 2: "negative"}

rows = []
for r, c in curve.items():
    last5 = c[c["round"] >= 10]
    rows.append({"annotator": r[0], "strategy": r[1],
                 "F1@1600": c.test_f1_macro.iloc[-1], "acc@1600": c.test_accuracy.iloc[-1],
                 "F1 mean r10-14": last5.test_f1_macro.mean(), "acc mean r10-14": last5.test_accuracy.mean(),
                 "F1 ALC (all rounds)": c.test_f1_macro.mean(),
                 "round-to-round sd (r8-14, F1)": c[c["round"] >= 8].test_f1_macro.diff().std(),
                 "first size F1>=0.65": next((int(t) for t, f in zip(c.train_size, c.test_f1_macro) if f >= 0.65), None),
                 "first size F1>=0.70": next((int(t) for t, f in zip(c.train_size, c.test_f1_macro) if f >= 0.70), None)})
print("== curve summary ==\n", pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== human vs LLM gap, mean of rounds 10-14 (points) ==")
for s in ("random", "margin", "entropy"):
    h, l = (curve[(a, s)].query("round>=10") for a in ("human", "llm"))
    print(f"{s:8s} macro-F1 gap {100*(h.test_f1_macro.mean()-l.test_f1_macro.mean()):.1f} | accuracy gap {100*(h.test_accuracy.mean()-l.test_accuracy.mean()):.1f}")

print("\n== AL minus random within each annotator, mean rounds 10-14 (macro-F1 points) ==")
for a in ("human", "llm"):
    base = curve[(a, "random")].query("round>=10").test_f1_macro.mean()
    for s in ("margin", "entropy"):
        print(f"{a:5s} {s:8s} {100*(curve[(a, s)].query('round>=10').test_f1_macro.mean()-base):+.1f}")

print("\n== LLM annotator agreement with gold on queried rows ==")
for r in runs[3:]:
    a = ann[r]; ag = a.true_label == a.train_label
    print(r[1], f"overall {ag.mean():.3f} (n={len(a)}) | by round:", np.round(ag.groupby(a['round']).mean().values, 2).tolist())

print("\n== neutral share of queried rows: gold vs LLM label ==")
for r in runs[3:]:
    a = ann[r]; print(r[1], f"gold {np.mean(a.true_label == 0):.3f}  LLM {np.mean(a.train_label == 0):.3f}")

print("\n== LLM annotator confusion (rows = gold, cols = LLM) ==")
for r in runs[3:]:
    a = ann[r]
    cm = pd.crosstab(a.true_label.map(names), a.train_label.map(names)).reindex(index=list(names.values()), columns=list(names.values()), fill_value=0)
    print(r[1]); print(cm)

print("\n== LLM confidence vs agreement (pooled over the 3 LLM runs) ==")
allc = pd.concat([ann[r] for r in runs[3:]])
bins = pd.cut(allc.annotator_conf, [0, .5, .7, .9, .99, 1.0001])
print(allc.assign(ok=allc.true_label == allc.train_label).groupby(bins, observed=True).ok.agg(["mean", "count"]).round(3))
print(f"mean confidence {allc.annotator_conf.mean():.3f} vs mean agreement {(allc.true_label == allc.train_label).mean():.3f}")

print("\n== mean sentence length (chars) of queried rows ==")
for r in runs: print(r, round(ann[r].n_chars.mean(), 1))

print("\n== overlap of queried rows between runs ==")
sets = {r: set(ann[r].sent_id) for r in runs}
for r1, r2 in [(("human", "margin"), ("human", "entropy")), (("human", "margin"), ("llm", "margin")),
               (("human", "entropy"), ("llm", "entropy")), (("human", "random"), ("llm", "random")),
               (("human", "margin"), ("human", "random"))]:
    print(r1, "vs", r2, ":", len(sets[r1] & sets[r2]), "of", len(sets[r1]))
