"""Which kinds of mistakes does the few-shot + calibration annotator make on the rows AL queried?

Pools the llm_fs margin and entropy runs of both seeds and prints, for the rows where the LLM label differs from gold,
the share of each (gold -> LLM) pair, plus the LLM confidence of those wrong rows.
Usage: python analysis/error_types.py --results results
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
D = ap.parse_args().results
NAMES = {0: "neutral", 1: "positive", 2: "negative"}
parts = [pd.read_csv(f"{D}/annotations_anon/sentigold_s{s}_llm_fs_{k}_annotations_anon.csv")
         for s in (17, 18) for k in ("margin", "entropy")]
A = pd.concat(parts, ignore_index=True)
A["gold"], A["llm"] = A.true_label.map(NAMES), A.train_label.map(NAMES)
W = A[A.true_label != A.train_label]
print(f"rows: {len(A)} | disagreements: {len(W)} ({len(W) / len(A):.1%})")
sh = (W.groupby(["gold", "llm"]).size() / len(W)).round(3).sort_values(ascending=False)
print("\nshare of all disagreements (gold -> LLM):"); print(sh.to_string())
neu = W[(W.gold == "neutral") | (W.llm == "neutral")]
print(f"\ndisagreements that involve neutral: {len(neu) / len(W):.1%}")
print(f"wrong rows with LLM confidence >= 0.95: {(W.annotator_conf >= 0.95).mean():.1%}")
