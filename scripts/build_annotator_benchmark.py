"""Step 1 of the annotator-improvement study: freeze a benchmark of rows to annotate and record the zero-shot baseline.

Runs in Colab (Drive mounted) after the three LLM-annotator runs have finished.
Reads  <CKPT_DIR>/sentigold_llm_annotator_{random,margin,entropy}_annotations.csv   (raw logs, contain sentences)
Writes <CKPT_DIR>/annotator_benchmark.csv                                          (kept OUT of git: contains sentences)
"""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CKPT_DIR = "/content/drive/MyDrive/llm_results"
N_PER_SOURCE, SEED = 600, 17

parts = []
for strat in ("random", "margin", "entropy"):
    a = pd.read_csv(f"{CKPT_DIR}/sentigold_llm_annotator_{strat}_annotations.csv")
    parts.append(a.sample(N_PER_SOURCE, random_state=SEED).assign(source=strat))
bench = pd.concat(parts, ignore_index=True).drop_duplicates(subset="sentence").reset_index(drop=True)  # a row queried by two strategies counts once
bench = bench.rename(columns={"true_label": "gold", "train_label": "zs_label", "annotator_conf": "zs_conf"})
bench = bench[["sentence", "gold", "zs_label", "zs_conf", "source"]]
bench.to_csv(f"{CKPT_DIR}/annotator_benchmark.csv", index=False)

print("benchmark rows:", len(bench), "| per source:", bench.source.value_counts().to_dict())
for src, g in [("ALL", bench)] + list(bench.groupby("source")):
    print(f"{src:8s} n={len(g):4d} acc={accuracy_score(g.gold, g.zs_label):.3f} "
          f"macroF1={f1_score(g.gold, g.zs_label, average='macro'):.3f} "
          f"LLM says neutral {np.mean(g.zs_label == 0):.0%} (gold {np.mean(g.gold == 0):.0%})")
