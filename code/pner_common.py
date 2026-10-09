"""
pner_common.py
==============
Shared code for ALL Pashto-NER experiments (ML, DL, transformers, hybrids).

Why this file exists (reviewer points 1, 2, 5, 6):
  * every model reads the SAME sentences / SAME split from work/data.json and work/splits.json
  * every model is scored with the SAME strict span-level evaluator (O is never counted)
  * every model writes its test predictions to work/preds/<model>__seed<k>.json, so all tables,
    figures and significance tests are produced from saved predictions (no typed-in numbers).
"""
import json
import os
import platform
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("PNER_WORK", "work"))
PREDS = WORK / "preds"
SEEDS = [42, 43, 44, 45, 46]          # 5 independent runs for every stochastic model

# id : (display name, family, input representation)
MODEL_INFO = OrderedDict([
    # ---- classical machine learning (hand-crafted features, window +-2) ----
    ("crf",             ("CRF",                    "ML",          "sparse features")),
    ("hmm",             ("HMM",                    "ML",          "word identity")),
    ("svm",             ("SVM",                    "ML",          "sparse features")),
    ("lr",              ("Logistic Regression",    "ML",          "sparse features")),
    ("rf",              ("Random Forest",          "ML",          "sparse features")),
    ("xgb",             ("XGBoost",                "ML",          "sparse features")),
    ("nb",              ("Naive Bayes",            "ML",          "sparse features")),
    # ---- neural sequence models (Word2Vec-initialised word embeddings) ----
    ("rnn",             ("RNN",                    "DL",          "word emb.")),
    ("lstm",            ("LSTM",                   "DL",          "word emb.")),
    ("bilstm",          ("Bi-LSTM",                "DL",          "word emb.")),
    ("gru",             ("GRU",                    "DL",          "word emb.")),
    ("bigru",           ("Bi-GRU",                 "DL",          "word emb.")),
    # ---- fine-tuned transformers (linear token-classification head) ----
    ("mbert",           ("mBERT",                  "Transformer", "subword")),
    ("distilmbert",     ("Distil-mBERT",           "Transformer", "subword")),
    ("xlmr",            ("XLM-R base",             "Transformer", "subword")),
    ("xlmr_weighted",   ("XLM-R base (weighted)",  "Transformer", "subword")),
    ("xlmr_large",      ("XLM-R large",            "Transformer", "subword")),
    # ---- hybrids ----
    ("bilstm_crf",      ("Bi-LSTM + CRF",          "Hybrid",      "word emb.")),
    ("cnn_bigru_attn",  ("CNN + Bi-GRU + Attn",    "Hybrid",      "word + char emb.")),
    ("xlmr_crf",        ("XLM-R base + CRF",       "Hybrid",      "subword")),
    ("xlmr_bilstm_crf", ("XLM-R base + Bi-LSTM + CRF", "Hybrid",  "subword")),
    ("xlmr_large_crf",  ("XLM-R large + CRF",      "Hybrid",      "subword")),
])


# --------------------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in ["numpy", "scikit_learn", "sklearn_crfsuite", "xgboost", "torch",
                "transformers", "gensim", "torchcrf"]:
        try:
            mod = __import__("sklearn" if lib == "scikit_learn" else lib)
            info[lib] = getattr(mod, "__version__", "n/a")
        except Exception:
            pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["gpu"] = "cpu"
    return info


# --------------------------------------------------------------------------------------
# data + split access (written by 01_prepare_data.py)
# --------------------------------------------------------------------------------------
def load_data():
    with open(WORK / "data.json", encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    with open(WORK / "splits.json", encoding="utf-8") as f:
        return json.load(f)


def get_split(name):
    """returns (sentences, tag_sequences) of split `name` in {train, val, test}."""
    d, s = load_data(), load_splits()
    idx = s[name]
    return [d["sentences"][i] for i in idx], [d["tags"][i] for i in idx]


def get_all_splits():
    d = load_data()
    out = {n: get_split(n) for n in ("train", "val", "test")}
    return d, out


# --------------------------------------------------------------------------------------
# STRICT span-level evaluation  (exact boundary + exact type, "O" never counted)
# --------------------------------------------------------------------------------------
def split_tag(t):
    if t == "O":
        return "O", None
    p, ty = t.split("-", 1)
    return p, ty


def extract_spans(tags):
    """Well-formed spans of a BIOES-style sequence:  S-X   |   B-X (I-X)* E-X.
    Ill-formed fragments (e.g. B-X followed by O, orphan E-X) produce NO span, exactly as in
    CoNLL strict evaluation, so a model cannot gain credit from malformed output."""
    spans, i, n = set(), 0, len(tags)
    while i < n:
        p, ty = split_tag(tags[i])
        if p == "S":
            spans.add((i, i, ty))
            i += 1
        elif p == "B":
            j = i + 1
            while j < n:
                pj, tj = split_tag(tags[j])
                if pj == "I" and tj == ty:
                    j += 1
                else:
                    break
            if j < n:
                pj, tj = split_tag(tags[j])
                if pj == "E" and tj == ty:
                    spans.add((i, j, ty))
                    i = j + 1
                    continue
            i += 1
        else:
            i += 1
    return spans


def sentence_counts(gold, pred, types):
    """array (n_sent, n_types, 3) with [tp, fp, fn] per sentence and entity type."""
    assert len(gold) == len(pred), "gold / prediction length mismatch"
    tix = {t: i for i, t in enumerate(types)}
    C = np.zeros((len(gold), len(types), 3), dtype=np.int32)
    for k, (g, p) in enumerate(zip(gold, pred)):
        assert len(g) == len(p), f"sentence {k}: {len(g)} gold tags vs {len(p)} predicted tags"
        gs, ps = extract_spans(g), extract_spans(p)
        for s in gs & ps:
            C[k, tix[s[2]], 0] += 1
        for s in ps - gs:
            if s[2] in tix:
                C[k, tix[s[2]], 1] += 1
        for s in gs - ps:
            C[k, tix[s[2]], 2] += 1
    return C


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def metrics_from_counts(C, types):
    tot = C.sum(axis=0)                       # (T, 3)
    per = OrderedDict()
    for i, t in enumerate(types):
        p, r, f = prf(*tot[i])
        per[t] = {"precision": p, "recall": r, "f1": f, "support": int(tot[i, 0] + tot[i, 2])}
    mp, mr, mf = prf(*tot.sum(axis=0))
    present = [per[t]["f1"] for t in types if per[t]["support"] > 0]
    return {"micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
            "macro_f1": float(np.mean(present)) if present else 0.0, "per_type": per}


def micro_f1(gold, pred, types):
    return prf(*sentence_counts(gold, pred, types).sum(axis=(0, 1)))[2]


# --------------------------------------------------------------------------------------
# prediction files
# --------------------------------------------------------------------------------------
def pred_path(model_id, seed):
    return PREDS / f"{model_id}__seed{seed}.json"


def already_done(model_id, seed):
    return pred_path(model_id, seed).exists()


def save_preds(model_id, seed, preds, meta):
    PREDS.mkdir(parents=True, exist_ok=True)
    preds = [[str(t) for t in seq] for seq in preds]          # plain python lists of str
    meta = dict(meta)
    meta["env"] = env_info()
    meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(pred_path(model_id, seed), "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "seed": seed, "meta": meta, "preds": preds}, f,
                  ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_preds(model_id, seed):
    with open(pred_path(model_id, seed), encoding="utf-8") as f:
        return json.load(f)


def available_seeds(model_id):
    return [s for s in SEEDS if already_done(model_id, s)]
