#!/usr/bin/env python
"""
02_ml_models.py -- classical models on the common split.

Protocol (identical for every model):
  * features are computed per SENTENCE, with a +-2 word window (no re-ordering, no O-downsampling)
  * hyper-parameters are chosen on the VALIDATION set (strict micro-F1), from the small grids below
  * the model is then trained on TRAIN only and predicts the TEST set once
  * stochastic models (RF, XGBoost) are repeated for 5 seeds; deterministic ones (CRF, HMM, SVM, LR,
    NB) are run once - their variability is covered by the bootstrap in 05_aggregate.py
  * the token classifiers (SVM, LR, RF, XGB, NB) predict every token independently; they have no
    sequence decoder, so malformed tag sequences simply earn no credit under strict evaluation

usage: python 02_ml_models.py [--models crf svm ...] [--xgb_device cpu|cuda]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pner_common import (SEEDS, already_done, get_all_splits, load_preds, micro_f1, save_preds,
                         set_seed)

ML_MODELS = ["crf", "hmm", "svm", "lr", "rf", "xgb", "nb"]
STOCHASTIC = {"rf", "xgb"}

GRIDS = {
    "crf": [dict(c1=a, c2=b) for a, b in itertools.product([0.05, 0.1, 0.3], [0.05, 0.1, 0.3])],
    "hmm": [dict(k_emit=k) for k in (0.001, 0.01, 0.1)],
    "svm": [dict(C=c, class_weight=w) for c, w in itertools.product([0.1, 1, 10], [None, "balanced"])],
    "lr":  [dict(C=c, class_weight=w) for c, w in itertools.product([1, 10], [None, "balanced"])],
    "rf":  [dict(class_weight=w) for w in (None, "balanced_subsample")],
    "xgb": [dict(max_depth=6)],
    "nb":  [dict(alpha=a) for a in (0.01, 0.1, 1.0)],
}


# ------------------------------------------------------------------ features
def shape(w):
    out = []
    for ch in w:
        c = "D" if ch.isdigit() else ("a" if ch.isalpha() else "p")
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def token_feats(ws, i):
    """case-free feature set (Pashto has no capitalisation): identity, affixes, shape, +-2 context."""
    w = ws[i]
    f = {"bias": 1.0, f"w={w}": 1.0, f"len={min(len(w), 12)}": 1.0, f"shape={shape(w)}": 1.0}
    for k in (1, 2, 3, 4):
        if len(w) >= k:
            f[f"pre{k}={w[:k]}"] = 1.0
            f[f"suf{k}={w[-k:]}"] = 1.0
    if any(ch.isdigit() for ch in w):
        f["has_digit"] = 1.0
    if all(ch.isdigit() for ch in w):
        f["all_digit"] = 1.0
    if any("a" <= ch.lower() <= "z" for ch in w):
        f["has_latin"] = 1.0
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(ws):
            f[f"w[{off}]={ws[j]}"] = 1.0
            if abs(off) == 1:
                f[f"suf2[{off}]={ws[j][-2:]}"] = 1.0
                f[f"suf3[{off}]={ws[j][-3:]}"] = 1.0
        else:
            f[f"w[{off}]=<pad>"] = 1.0
    if i > 0:
        f[f"w[-1]|w={ws[i-1]}|{w}"] = 1.0
    if i < len(ws) - 1:
        f[f"w|w[+1]={w}|{ws[i+1]}"] = 1.0
    return f


def sent_feats(ws):
    return [token_feats(ws, i) for i in range(len(ws))]


def unflatten(flat, sents):
    out, k = [], 0
    for s in sents:
        out.append(list(flat[k:k + len(s)]))
        k += len(s)
    return out


# ------------------------------------------------------------------ HMM (supervised, 1st order)
class HMM:
    def __init__(self, labels, k_emit=0.01, k_trans=0.1, min_count=2):
        self.labels, self.k_emit, self.k_trans, self.min_count = labels, k_emit, k_trans, min_count
        self.l2i = {l: i for i, l in enumerate(labels)}

    def fit(self, sents, tags):
        from collections import Counter
        cnt = Counter(w for s in sents for w in s)
        self.vocab = {w: i for i, w in enumerate(w for w, c in cnt.items() if c >= self.min_count)}
        V, L = len(self.vocab) + 1, len(self.labels)            # last column = <UNK>
        E, T, S = np.zeros((L, V)), np.zeros((L, L)), np.zeros(L)
        for s, ts in zip(sents, tags):
            prev = None
            for w, t in zip(s, ts):
                li = self.l2i[t]
                E[li, self.vocab.get(w, V - 1)] += 1
                if prev is None:
                    S[li] += 1
                else:
                    T[prev, li] += 1
                prev = li
        self.logE = np.log((E + self.k_emit) / (E.sum(1, keepdims=True) + self.k_emit * V))
        self.logT = np.log((T + self.k_trans) / (T.sum(1, keepdims=True) + self.k_trans * L))
        self.logS = np.log((S + self.k_trans) / (S.sum() + self.k_trans * L))
        return self

    def predict(self, sents):
        V = len(self.vocab) + 1
        out = []
        for s in sents:
            ids = [self.vocab.get(w, V - 1) for w in s]
            n, L = len(ids), len(self.labels)
            delta = np.zeros((n, L))
            back = np.zeros((n, L), dtype=int)
            delta[0] = self.logS + self.logE[:, ids[0]]
            for t in range(1, n):
                sc = delta[t - 1][:, None] + self.logT
                back[t] = sc.argmax(0)
                delta[t] = sc.max(0) + self.logE[:, ids[t]]
            path = [int(delta[-1].argmax())]
            for t in range(n - 1, 0, -1):
                path.append(int(back[t][path[-1]]))
            out.append([self.labels[i] for i in reversed(path)])
        return out


# ------------------------------------------------------------------ one fit -> predictions
def fit_predict(kind, cfg, seed, D, eval_names, args):
    """D holds train/val/test sentences, tags, features. returns {name: predicted tag sequences}."""
    labels = D["labels"]
    tr_s, tr_t = D["train"]
    out = {}
    if kind == "crf":
        import sklearn_crfsuite
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=cfg["c1"], c2=cfg["c2"], max_iterations=150,
                                   all_possible_transitions=True)
        crf.fit(D["feats"]["train"], tr_t)
        for n in eval_names:
            out[n] = crf.predict(D["feats"][n])
        return out
    if kind == "hmm":
        h = HMM(labels, k_emit=cfg["k_emit"]).fit(tr_s, tr_t)
        for n in eval_names:
            out[n] = h.predict(D[n][0])
        return out

    # ---- independent token classifiers on sparse features
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder
    if "vec" not in D:
        vec = DictVectorizer(sparse=True, dtype=np.float32)
        flat = lambda n: [f for s in D["feats"][n] for f in s]
        Xtr = vec.fit_transform(flat("train"))
        keep = np.where(np.asarray((Xtr != 0).sum(axis=0)).ravel() >= 2)[0]   # min. doc. freq 2
        D["vec"] = (vec, keep)
        D["X"] = {"train": Xtr[:, keep]}
        for n in ("val", "test"):
            D["X"][n] = vec.transform(flat(n))[:, keep]
        D["y"] = {n: np.array([t for ts in D[n][1] for t in ts]) for n in ("train", "val", "test")}
    Xtr, ytr = D["X"]["train"], D["y"]["train"]
    le = LabelEncoder().fit(ytr)
    ytr_i = le.transform(ytr)

    if kind == "svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=5000, random_state=0)
    elif kind == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=300)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_trees, max_features="sqrt",
                                     class_weight=cfg["class_weight"], n_jobs=-1, random_state=seed)
    elif kind == "nb":
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=cfg["alpha"])
    elif kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=cfg["max_depth"],
                            subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                            device=args.xgb_device, early_stopping_rounds=20, eval_metric="mlogloss",
                            random_state=seed, n_jobs=-1)
    if kind == "xgb":
        yva = D["y"]["val"]
        ok = np.isin(yva, le.classes_)
        clf.fit(Xtr, ytr_i, eval_set=[(D["X"]["val"][ok], le.transform(yva[ok]))], verbose=False)
    else:
        clf.fit(Xtr, ytr_i)
    for n in eval_names:
        out[n] = unflatten(le.inverse_transform(clf.predict(D["X"][n])), D[n][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ML_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--rf_trees", type=int, default=200)
    ap.add_argument("--xgb_device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    data, sp = get_all_splits()
    types, labels = data["types"], data["labels"]
    D = {"labels": labels, **sp, "feats": {n: [sent_feats(s) for s in sp[n][0]] for n in sp}}

    for kind in args.models:
        seeds = args.seeds if kind in STOCHASTIC else args.seeds[:1]
        if all(already_done(kind, s) for s in seeds):
            print(f"[{kind}] already done - skipping")
            continue
        # ---- model selection on VALIDATION (first seed only)
        t0, best_cfg, best_f1 = time.time(), None, -1
        if already_done(kind, seeds[0]):
            # extra seeds are being added later: reuse the configuration chosen for the first seed
            prev = load_preds(kind, seeds[0])["meta"]
            best_cfg, best_f1 = prev["config"], prev["val_micro_f1"]
            print(f"[{kind}] re-using configuration from seed {seeds[0]}: {best_cfg}")
        else:
            for cfg in GRIDS[kind]:
                set_seed(seeds[0])
                pred = fit_predict(kind, cfg, seeds[0], D, ["val"], args)["val"]
                f1 = micro_f1(sp["val"][1], pred, types)
                print(f"[{kind}] {cfg} -> val micro-F1 {f1:.4f}")
                if f1 > best_f1:
                    best_cfg, best_f1 = cfg, f1
            print(f"[{kind}] selected {best_cfg} (val micro-F1 {best_f1:.4f})")
        # ---- final fit on TRAIN, single look at TEST
        for seed in seeds:
            if already_done(kind, seed):
                continue
            set_seed(seed)
            t1 = time.time()
            pred = fit_predict(kind, best_cfg, seed, D, ["test"], args)["test"]
            test_f1 = micro_f1(sp["test"][1], pred, types)
            save_preds(kind, seed, pred, {
                "config": best_cfg, "val_micro_f1": best_f1, "family": "ML",
                "train_seconds": round(time.time() - t1, 1),
                "n_train_sentences": len(sp["train"][0]), "deterministic": kind not in STOCHASTIC})
            print(f"[{kind}] seed {seed}: test strict micro-F1 {test_f1:.4f}")
        print(f"[{kind}] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
