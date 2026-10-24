"""Turn the raw annotation logs (which contain dataset sentences) into shareable files.

The raw logs come from the AL notebook (<run>_annotations.csv). The shareable version drops the sentence text and keeps
  sent_id  = first 12 hex chars of sha1(sentence in UTF-8)  -> lets anyone who has the dataset re-link rows
  n_chars  = sentence length in characters
Usage: python scripts/anonymize_logs.py RAW_DIR OUT_DIR
"""
import hashlib, sys, glob, os
import pandas as pd

raw_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

def sid(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

for path in sorted(glob.glob(f"{raw_dir}/sentigold_*_annotations.csv")):
    name = os.path.basename(path)                                   # sentigold_<annotator>_annotator_<strategy>_annotations.csv
    _, annotator, _, strategy, _ = name[:-4].split("_")
    d = pd.read_csv(path)
    d["sent_id"] = d["sentence"].map(sid)
    assert d["sent_id"].nunique() == d["sentence"].nunique(), "hash collision"
    d["n_chars"] = d["sentence"].str.len()
    d.insert(0, "strategy", strategy); d.insert(1, "annotator", annotator)
    d.drop(columns=["sentence"]).to_csv(f"{out_dir}/{name.replace('_annotations.csv', '_annotations_anon.csv')}", index=False)
    print("wrote", name, len(d))
