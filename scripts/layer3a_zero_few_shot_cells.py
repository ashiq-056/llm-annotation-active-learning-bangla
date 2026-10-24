# ===== Layer 3a: zero/few-shot LLM baseline (paste each block as a Colab cell) =====
# Prereq: data cells (test_df, seed_train), model-loading cell (llm_tokenizer, llm_model), LABEL_MAP/LABEL_NAMES.

# ---- CELL A: batched logit-scoring classifier ----
import numpy as np, torch, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

llm_tokenizer.padding_side = "left"            # REQUIRED for decoder-only batching
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

# first token of each label word; order = [neutral(0), positive(1), negative(2)]
LABEL_WORDS = [LABEL_NAMES[i] for i in range(3)]
LABEL_TOKEN_IDS = [llm_tokenizer.encode(w, add_special_tokens=False)[0] for w in LABEL_WORDS]
print(dict(zip(LABEL_WORDS, LABEL_TOKEN_IDS)), [llm_tokenizer.decode([t]) for t in LABEL_TOKEN_IDS])
assert len(set(LABEL_TOKEN_IDS)) == 3, "label first-tokens collide; use a different label word"

def make_chat_text(sentence, examples=None):
    msgs = [{"role": "user", "content": build_prompt(sentence, examples)}]
    return llm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def llm_label_probs(sentences, examples=None, batch_size=16, verbose=True):
    """Returns (n,3) probs over [neutral, positive, negative], renormalised over the 3 label tokens."""
    texts = [make_chat_text(s, examples) for s in sentences]
    order = np.argsort([len(t) for t in texts])          # length-sort -> less padding waste
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i+batch_size]
        enc = llm_tokenizer([texts[j] for j in idx], return_tensors="pt", padding=True).to(llm_model.device)
        logits = llm_model(**enc).logits[:, -1, :]       # last position (left padding => aligned)
        sel = logits[:, LABEL_TOKEN_IDS].float()
        out[idx] = torch.softmax(sel, dim=-1).cpu().numpy()
        if verbose and (i // batch_size) % 10 == 0:
            print(f"{min(i+batch_size, len(texts))}/{len(texts)}")
    return out

# ---- CELL B: sanity check on 40 rows (also verify the label mapping by eye!) ----
sub = test_df.sample(40, random_state=0)
p = llm_label_probs(sub["sentence"].tolist())
sub = sub.assign(pred=p.argmax(1))
for _, r in sub.head(15).iterrows():
    print(f"true={LABEL_NAMES[r.sentiment]:8s} pred={LABEL_NAMES[r.pred]:8s} | {r.sentence[:70]}")
print("acc on 40:", (sub.pred == sub.sentiment).mean())

# ---- CELL C: zero-shot on the full locked test set ----
probs_zs = llm_label_probs(test_df["sentence"].tolist(), examples=None, batch_size=16)
preds_zs = probs_zs.argmax(1)
y = test_df["sentiment"].values
zs_f1, zs_acc = f1_score(y, preds_zs, average="macro"), accuracy_score(y, preds_zs)
print(f"Zero-shot Qwen2.5-7B: macro-F1={zs_f1:.3f} acc={zs_acc:.3f}")
print(classification_report(y, preds_zs, target_names=LABEL_WORDS, digits=3))
print(confusion_matrix(y, preds_zs))

# ---- CELL D: few-shot (k per class from the 200-row seed set) ----
def pick_shots(k_per_class=4, seed=17):
    shots = []
    for lab in range(3):
        rows = seed_train[seed_train.sentiment == lab].sample(k_per_class, random_state=seed)
        shots += list(zip(rows.sentence, rows.sentiment))
    rng = np.random.default_rng(seed); rng.shuffle(shots)
    return shots

shots = pick_shots(4)
probs_fs = llm_label_probs(test_df["sentence"].tolist(), examples=shots, batch_size=8)  # longer prompts -> smaller batch
preds_fs = probs_fs.argmax(1)
fs_f1, fs_acc = f1_score(y, preds_fs, average="macro"), accuracy_score(y, preds_fs)
print(f"Few-shot (4/class): macro-F1={fs_f1:.3f} acc={fs_acc:.3f}")

# ---- CELL E: save results + disagreements for manual inspection ----
import pickle
from google.colab import files as colab_files
with open("llm_zeroshot_result.pkl", "wb") as f:
    pickle.dump({"zs_f1": zs_f1, "zs_acc": zs_acc, "fs_f1": fs_f1, "fs_acc": fs_acc,
                 "probs_zs": probs_zs, "probs_fs": probs_fs}, f)
dis = test_df.assign(true=[LABEL_NAMES[i] for i in y],
                     zs_pred=[LABEL_NAMES[i] for i in preds_zs],
                     zs_conf=probs_zs.max(1))
dis = dis[dis.true != dis.zs_pred].sample(frac=1, random_state=0)
dis.head(60).to_csv("zs_disagreements.csv", index=False)
colab_files.download("llm_zeroshot_result.pkl")
colab_files.download("zs_disagreements.csv")
