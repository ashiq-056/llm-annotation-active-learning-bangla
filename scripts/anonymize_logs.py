"""Turn the raw files from Drive (which contain dataset sentences) into shareable files.

Usage: python scripts/anonymize_logs.py RAW_DIR results

Reads from RAW_DIR:
  sentigold[_sNN]_<annotator>_annotator_<strategy>_annotations.csv   (annotator: human, llm, llm_fs)
  sentigold[_sNN]_<annotator>_annotator_<strategy>_curve.csv
  annotator_benchmark.csv, step5_annotator*.json, step2_offsets.json, step4_chosen_k.json, step2d_predictions.csv
Writes to OUT:
  curves/, annotations_anon/, benchmark/annotator_benchmark_anon.csv, annotator/*.json

Every sentence is replaced by
  sent_id = first 12 hex chars of sha1(sentence in UTF-8)   (lets anyone who has the dataset re-link rows)
  n_chars = sentence length in characters
"""
import glob, hashlib, json, os, re, shutil, sys
import pandas as pd

raw, out = sys.argv[1], sys.argv[2]
for d in ("curves", "annotations_anon", "annotator", "benchmark"):
    os.makedirs(f"{out}/{d}", exist_ok=True)

def sid(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

PAT = re.compile(r"^sentigold(?:_s(\d+))?_(human|llm_fs|llm)_annotator_(balanced_random|random|margin|entropy)_(annotations|curve)\.csv$")

for path in sorted(glob.glob(f"{raw}/sentigold*_annotator_*.csv")):
    name = os.path.basename(path)
    m = PAT.match(name)
    assert m, f"unexpected file name: {name}"
    seed = int(m.group(1)) if m.group(1) else 17           # files without _sNN are seed 17
    annotator, strategy, kind = m.group(2), m.group(3), m.group(4)
    tag = f"sentigold_s{seed}_{annotator}_{strategy}"
    if kind == "curve":
        shutil.copy(path, f"{out}/curves/{tag}_curve.csv")
        continue
    d = pd.read_csv(path)
    d["sent_id"] = d["sentence"].map(sid)
    assert d["sent_id"].nunique() == d["sentence"].nunique(), "hash collision"
    d["n_chars"] = d["sentence"].str.len()
    d.insert(0, "seed", seed); d.insert(1, "annotator", annotator); d.insert(2, "strategy", strategy)
    d.drop(columns=["sentence"]).to_csv(f"{out}/annotations_anon/{tag}_annotations_anon.csv", index=False)

b = pd.read_csv(f"{raw}/annotator_benchmark.csv")
b.insert(0, "sent_id", b["sentence"].map(sid)); b.insert(1, "n_chars", b["sentence"].str.len())
b.drop(columns=["sentence"]).to_csv(f"{out}/benchmark/annotator_benchmark_anon.csv", index=False)

shutil.copy(f"{raw}/step2d_predictions.csv", f"{out}/benchmark/zero_shot_calibration_predictions.csv")
for name in ("step2_offsets.json", "step4_chosen_k.json"):
    shutil.copy(f"{raw}/{name}", f"{out}/annotator/{name}")
for path in sorted(glob.glob(f"{raw}/step5_annotator*.json")):
    j = json.load(open(path))
    shots = j.get("shots", [])
    j["shots"] = [{"sent_id": sid(s), "label": int(l)} for s, l in shots]     # sentences removed, ids kept
    json.dump(j, open(f"{out}/annotator/{os.path.basename(path)}", "w"), indent=1)
print("done:", {d: len(os.listdir(f"{out}/{d}")) for d in ("curves", "annotations_anon", "annotator", "benchmark")})
