#!/usr/bin/env python
"""
03_dl_models.py -- neural sequence taggers on the common split (PyTorch).

Common protocol for all 7 models below
  * word embeddings (128-d) initialised with Word2Vec trained on the TRAIN sentences only
    (falls back to random init if gensim is missing), fine-tuned during training
  * no sentence truncation (batches are padded to the longest sentence; RNNs use packed sequences)
  * Adam, lr 1e-3, batch 32, dropout 0.3, grad-clip 5, hidden size 128 per direction
  * max 40 epochs (at least 10), early stopping on VALIDATION strict micro-F1 (patience 6); best epoch restored
  * 5 seeds (42..46); test set is predicted once, after the best epoch is restored

models   rnn | lstm | bilstm | gru | bigru                (plain softmax output)
         bilstm_crf                                       (Bi-LSTM + CRF output layer)
         cnn_bigru_attn                                   (char-CNN + Bi-GRU + self-attention)

usage:   python 03_dl_models.py [--models bilstm gru] [--seeds 42 43] [--max_epochs 40]
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

sys.path.insert(0, str(Path(__file__).parent))
from pner_common import SEEDS, already_done, get_all_splits, micro_f1, save_preds, set_seed

CFG = {
    "rnn":            dict(rnn="rnn",  bi=False, char=False, attn=False, crf=False),
    "lstm":           dict(rnn="lstm", bi=False, char=False, attn=False, crf=False),
    "bilstm":         dict(rnn="lstm", bi=True,  char=False, attn=False, crf=False),
    "gru":            dict(rnn="gru",  bi=False, char=False, attn=False, crf=False),
    "bigru":          dict(rnn="gru",  bi=True,  char=False, attn=False, crf=False),
    "bilstm_crf":     dict(rnn="lstm", bi=True,  char=False, attn=False, crf=True),
    "cnn_bigru_attn": dict(rnn="gru",  bi=True,  char=True,  attn=True,  crf=False),
}
HP = dict(emb_dim=128, hidden=128, dropout=0.3, lr=1e-3, batch_size=32, clip=5.0,
          max_epochs=40, min_epochs=10, patience=6, word_dropout=0.05, char_dim=30, char_filters=50,
          max_word_chars=20, attn_heads=4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ vocabularies / tensors
class Vocab:
    def __init__(self, sents, labels):
        words = sorted({w for s in sents for w in s})
        chars = sorted({c for w in words for c in w})
        self.w2i = {w: i + 2 for i, w in enumerate(words)}            # 0 = PAD, 1 = UNK
        self.c2i = {c: i + 2 for i, c in enumerate(chars)}
        self.l2i = {l: i for i, l in enumerate(labels)}
        self.labels = labels


def make_batch(sents, tags, idx, V, use_char, train=False, word_dropout=0.0):
    B = len(idx)
    L = max(len(sents[i]) for i in idx)
    w = torch.zeros(B, L, dtype=torch.long)
    y = torch.full((B, L), -100, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    c = torch.zeros(B, L, HP["max_word_chars"], dtype=torch.long) if use_char else None
    for b, i in enumerate(idx):
        s = sents[i]
        lens[b] = len(s)
        for j, tok in enumerate(s):
            w[b, j] = V.w2i.get(tok, 1)
            if use_char:
                for k, ch in enumerate(tok[:HP["max_word_chars"]]):
                    c[b, j, k] = V.c2i.get(ch, 1)
        if tags is not None:
            for j, t in enumerate(tags[i]):
                y[b, j] = V.l2i[t]
    if train and word_dropout > 0:                                    # randomly replace words by UNK
        drop = (torch.rand(w.shape) < word_dropout) & (w > 0)
        w = w.masked_fill(drop, 1)
    return w.to(dev), (c.to(dev) if c is not None else None), y.to(dev), lens


def word2vec_matrix(train_sents, V, dim, seed):
    mat = np.random.RandomState(seed).normal(0, 0.1, (len(V.w2i) + 2, dim)).astype(np.float32)
    mat[0] = 0
    try:
        from gensim.models import Word2Vec
    except ImportError:
        return torch.tensor(mat), False
    w2v = Word2Vec(sentences=train_sents, vector_size=dim, window=5, min_count=1, sg=1,
                   epochs=30, seed=seed, workers=1)
    for w, i in V.w2i.items():
        if w in w2v.wv:
            mat[i] = w2v.wv[w]
    return torch.tensor(mat), True


# ------------------------------------------------------------------ model
class Tagger(nn.Module):
    def __init__(self, n_words, n_chars, n_labels, cfg, emb_init=None):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(n_words, HP["emb_dim"], padding_idx=0)
        if emb_init is not None:
            self.emb.weight.data.copy_(emb_init)
        d_in = HP["emb_dim"]
        if cfg["char"]:
            self.cemb = nn.Embedding(n_chars, HP["char_dim"], padding_idx=0)
            self.cconv = nn.Conv1d(HP["char_dim"], HP["char_filters"], 3, padding=1)
            d_in += HP["char_filters"]
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cfg["rnn"]]
        self.rnn = rnn_cls(d_in, HP["hidden"], batch_first=True, bidirectional=cfg["bi"])
        d_out = HP["hidden"] * (2 if cfg["bi"] else 1)
        self.drop = nn.Dropout(HP["dropout"])
        if cfg["attn"]:
            self.attn = nn.MultiheadAttention(d_out, HP["attn_heads"], dropout=HP["dropout"],
                                              batch_first=True)
            self.ln = nn.LayerNorm(d_out)
        self.fc = nn.Linear(d_out, n_labels)
        self.crf = CRF(n_labels, batch_first=True) if cfg["crf"] else None

    def emissions(self, w, c, lens):
        x = self.emb(w)
        if self.cfg["char"]:
            B, L, C = c.shape
            ce = self.cemb(c.view(B * L, C)).transpose(1, 2)           # (B*L, char_dim, C)
            ce = torch.relu(self.cconv(ce)).max(dim=2).values.view(B, L, -1)
            x = torch.cat([x, ce], dim=-1)
        x = self.drop(x)
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=w.size(1))
        if self.cfg["attn"]:
            a, _ = self.attn(h, h, h, key_padding_mask=(w == 0), need_weights=False)
            h = self.ln(h + self.drop(a))
        return self.fc(self.drop(h))

    def loss(self, w, c, y, lens):
        em = self.emissions(w, c, lens)
        if self.crf is not None:
            mask = w != 0
            return -self.crf(em, y.clamp(min=0), mask=mask, reduction="mean")
        return nn.functional.cross_entropy(em.view(-1, em.size(-1)), y.view(-1), ignore_index=-100)

    @torch.no_grad()
    def decode(self, w, c, lens):
        em = self.emissions(w, c, lens)
        if self.crf is not None:
            return self.crf.decode(em, mask=(w != 0))
        pred = em.argmax(-1).cpu().tolist()
        return [pred[b][:int(lens[b])] for b in range(len(pred))]


@torch.no_grad()
def predict(model, sents, V, use_char, bs=64):
    model.eval()
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    out = [None] * len(sents)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        w, c, _, lens = make_batch(sents, None, idx, V, use_char)
        for i, p in zip(idx, model.decode(w, c, lens)):
            out[i] = [V.labels[j] for j in p]
    return out


# ------------------------------------------------------------------ one run
def run(model_id, seed, data, sp, args):
    cfg = CFG[model_id]
    types, labels = data["types"], data["labels"]
    tr_s, tr_t = sp["train"]
    va_s, va_t = sp["val"]
    te_s, te_t = sp["test"]
    set_seed(seed)
    V = Vocab(tr_s, labels)
    emb, used_w2v = word2vec_matrix(tr_s, V, HP["emb_dim"], seed)
    model = Tagger(len(V.w2i) + 2, len(V.c2i) + 2, len(labels), cfg, emb).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    rng = np.random.RandomState(seed)

    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.max_epochs + 1):
        model.train()
        perm = rng.permutation(len(tr_s))
        for k in range(0, len(perm), HP["batch_size"]):
            idx = perm[k:k + HP["batch_size"]].tolist()
            w, c, y, lens = make_batch(tr_s, tr_t, idx, V, cfg["char"], train=True,
                                       word_dropout=HP["word_dropout"])
            opt.zero_grad()
            loss = model.loss(w, c, y, lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            opt.step()
        f1 = micro_f1(va_t, predict(model, va_s, V, cfg["char"]), types)
        if f1 > best_f1:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"  [{model_id} s{seed}] epoch {ep:02d} loss {loss.item():.4f} val micro-F1 {f1:.4f}"
              f"{'  *' if bad == 0 else ''}")
        if bad >= HP["patience"] and ep >= args.min_epochs:    # never stop before min_epochs
            break
    model.load_state_dict(best_state)
    pred = predict(model, te_s, V, cfg["char"])
    test_f1 = micro_f1(te_t, pred, types)
    n_par = sum(p.numel() for p in model.parameters())
    save_preds(model_id, seed, pred, {
        "config": {**HP, **cfg, "word2vec_init": used_w2v, "max_epochs": args.max_epochs,
                   "selection": "val strict micro-F1", "n_params": n_par},
        "val_micro_f1": best_f1, "best_epoch": best_ep, "train_seconds": round(time.time() - t0, 1),
        "family": "Hybrid" if (cfg["crf"] or cfg["attn"]) else "DL"})
    print(f"[{model_id}] seed {seed}: best epoch {best_ep}, val {best_f1:.4f}, test strict micro-F1 {test_f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CFG))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max_epochs", type=int, default=HP["max_epochs"])
    ap.add_argument("--min_epochs", type=int, default=HP["min_epochs"],
                    help="early stopping is not allowed before this epoch (val F1 is exactly 0 while "
                         "a model still predicts only 'O')")
    args = ap.parse_args()
    data, sp = get_all_splits()
    print("device:", dev)
    for m in args.models:
        for s in args.seeds:
            if already_done(m, s):
                print(f"[{m}] seed {s} already done - skipping")
                continue
            run(m, s, data, sp, args)


if __name__ == "__main__":
    main()
