#!/usr/bin/env python
"""
04_transformers.py -- ALL transformer experiments are FULL FINE-TUNING (no frozen encoders).

  mbert, distilmbert, xlmr, xlmr_weighted, xlmr_large      linear token-classification head
  xlmr_crf, xlmr_bilstm_crf, xlmr_large_crf                CRF / Bi-LSTM+CRF head on top of the encoder

Implementation details (reviewer points 5, 8, 9)
  * label alignment: the FIRST sub-token of each word carries the word's representation and label;
    other sub-tokens are ignored. Predictions are therefore already at WORD level, so the strict span
    evaluation is identical to the one used for the ML / DL models.
  * the CRF / Bi-LSTM operate on the word-level sequence (first sub-tokens), never on sub-words.
  * truncation at 256 sub-tokens; words beyond that are predicted "O" (counted as errors, not dropped).
  * AdamW, linear warm-up (10 %) + linear decay, weight decay 0.01, grad-clip 1.0, fp16 autocast on GPU.
    encoder lr: 3e-5 (base models) / 1e-5 (large);  head lr (linear / CRF / LSTM): 1e-3.
  * batch 16 (base) / 8 (large); max 15 epochs (at least 3); early stopping on VALIDATION strict micro-F1
    (patience 4); the best epoch is restored; the TEST set is predicted once.
  * "xlmr_weighted": class-weighted cross-entropy, w_c = min(10, sqrt(N / (K * n_c))) from TRAIN counts.
  * 5 seeds (42..46) for every model.

usage: python 04_transformers.py --models xlmr xlmr_crf --seeds 42 43
       python 04_transformers.py --models xlmr --model_path /path/to/local/model   (offline use)
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from pner_common import SEEDS, already_done, get_all_splits, micro_f1, save_preds, set_seed

BASE = dict(lr=3e-5, batch_size=16)
LARGE = dict(lr=1e-5, batch_size=8)
MODELS = {
    "mbert":           dict(path="bert-base-multilingual-cased",       head="linear", weighted=False, **BASE),
    "distilmbert":     dict(path="distilbert-base-multilingual-cased", head="linear", weighted=False, **BASE),
    "xlmr":            dict(path="xlm-roberta-base",                   head="linear", weighted=False, **BASE),
    "xlmr_weighted":   dict(path="xlm-roberta-base",                   head="linear", weighted=True,  **BASE),
    "xlmr_large":      dict(path="xlm-roberta-large",                  head="linear", weighted=False, **LARGE),
    "xlmr_crf":        dict(path="xlm-roberta-base",                   head="crf",    weighted=False, **BASE),
    "xlmr_bilstm_crf": dict(path="xlm-roberta-base",                   head="bilstm_crf", weighted=False, **BASE),
    "xlmr_large_crf":  dict(path="xlm-roberta-large",                  head="crf",    weighted=False, **LARGE),
}
HP = dict(max_len=256, max_epochs=15, min_epochs=3, patience=4, warmup=0.1, weight_decay=0.01, clip=1.0,
          head_lr=1e-3, dropout=0.1, lstm_hidden=256)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"


# ------------------------------------------------------------------ encoding
def encode(tok, sents, tags, l2i):
    """returns per sentence: input_ids, first_pos (position of first sub-token of each kept word),
    labels of kept words, and the ORIGINAL word index of each kept word."""
    enc = tok(sents, is_split_into_words=True, truncation=True, max_length=HP["max_len"])
    items = []
    for i in range(len(sents)):
        wids, first, kept, prev = enc.word_ids(i), [], [], None
        for pos, w in enumerate(wids):
            if w is not None and w != prev:
                first.append(pos)
                kept.append(w)
            prev = w
        lab = [l2i[tags[i][w]] for w in kept] if tags is not None else [0] * len(kept)
        items.append(dict(ids=enc["input_ids"][i], first=first, kept=kept, lab=lab))
    return items


def collate(items, idx, pad_id):
    B = len(idx)
    Lt = max(len(items[i]["ids"]) for i in idx)
    Lw = max(len(items[i]["first"]) for i in idx)
    ids = torch.full((B, Lt), pad_id, dtype=torch.long)
    am = torch.zeros(B, Lt, dtype=torch.long)
    first = torch.zeros(B, Lw, dtype=torch.long)
    wm = torch.zeros(B, Lw, dtype=torch.bool)
    lab = torch.full((B, Lw), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        it = items[i]
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"])
        am[b, :len(it["ids"])] = 1
        n = len(it["first"])
        first[b, :n] = torch.tensor(it["first"])
        wm[b, :n] = True
        lab[b, :n] = torch.tensor(it["lab"])
    return ids.to(dev), am.to(dev), first.to(dev), wm.to(dev), lab.to(dev)


# ------------------------------------------------------------------ model
class TransTagger(nn.Module):
    def __init__(self, path, n_labels, head, class_weights=None):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path)
        H = self.enc.config.hidden_size
        self.head = head
        self.drop = nn.Dropout(HP["dropout"])
        if head == "bilstm_crf":
            self.lstm = nn.LSTM(H, HP["lstm_hidden"], batch_first=True, bidirectional=True)
            H = 2 * HP["lstm_hidden"]
        self.fc = nn.Linear(H, n_labels)
        self.crf = CRF(n_labels, batch_first=True) if head in ("crf", "bilstm_crf") else None
        self.register_buffer("cw", class_weights if class_weights is not None else torch.empty(0))

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("enc.")]

    def emissions(self, ids, am, first, wm):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        h = h.float()                                                  # head + CRF in fp32
        h = h.gather(1, first.unsqueeze(-1).expand(-1, -1, h.size(-1)))   # word-level states
        h = self.drop(h)
        if self.head == "bilstm_crf":
            lens = wm.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=wm.size(1))
            h = self.drop(h)
        return self.fc(h)

    def loss(self, ids, am, first, wm, lab):
        em = self.emissions(ids, am, first, wm)
        if self.crf is not None:
            return -self.crf(em, lab.clamp(min=0), mask=wm, reduction="mean")
        w = self.cw if self.cw.numel() else None
        return nn.functional.cross_entropy(em.view(-1, em.size(-1)), lab.view(-1),
                                           weight=w, ignore_index=-100)

    @torch.no_grad()
    def decode(self, ids, am, first, wm):
        em = self.emissions(ids, am, first, wm)
        if self.crf is not None:
            return self.crf.decode(em, mask=wm)
        pred = em.argmax(-1).cpu().tolist()
        n = wm.sum(1).tolist()
        return [pred[b][:n[b]] for b in range(len(pred))]


@torch.no_grad()
def predict(model, items, sents, labels, pad_id, bs=32):
    """word-level label sequences; words lost by truncation are predicted 'O'."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [["O"] * len(s) for s in sents]
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        ids, am, first, wm, _ = collate(items, idx, pad_id)
        for i, p in zip(idx, model.decode(ids, am, first, wm)):
            for w, lab in zip(items[i]["kept"], p):
                out[i][w] = labels[lab]
    return out


# ------------------------------------------------------------------ one run
def run(model_id, seed, data, sp, args):
    spec = MODELS[model_id]
    path = args.model_path or spec["path"]
    lr = args.lr or spec["lr"]
    bs = args.batch_size or spec["batch_size"]
    labels, types = data["labels"], data["types"]
    l2i = {l: i for i, l in enumerate(labels)}
    tr_s, tr_t = sp["train"]
    va_s, va_t = sp["val"]
    te_s, te_t = sp["test"]

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(path)
    tr = encode(tok, tr_s, tr_t, l2i)
    va = encode(tok, va_s, va_t, l2i)
    te = encode(tok, te_s, te_t, l2i)
    n_trunc = sum(len(s) - len(it["kept"]) for s, it in zip(te_s, te))
    pad_id = tok.pad_token_id

    cw = None
    if spec["weighted"]:
        cnt = Counter(t for ts in tr_t for t in ts)
        N, K = sum(cnt.values()), len(labels)
        cw = torch.tensor([min(10.0, (N / (K * max(cnt.get(l, 0), 1))) ** 0.5) for l in labels],
                          dtype=torch.float)
    model = TransTagger(path, len(labels), spec["head"], cw).to(dev)

    nd = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    enc_params = [(n, p) for n, p in model.named_parameters() if n.startswith("enc.")]
    groups = [
        {"params": [p for n, p in enc_params if not any(x in n for x in nd)],
         "lr": lr, "weight_decay": HP["weight_decay"]},
        {"params": [p for n, p in enc_params if any(x in n for x in nd)], "lr": lr, "weight_decay": 0.0},
        {"params": model.head_params(), "lr": HP["head_lr"], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups)
    steps = args.max_epochs * int(np.ceil(len(tr) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(HP["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.RandomState(seed)

    best_f1, best_state, best_ep, bad, t0 = -1.0, None, 0, 0, time.time()
    for ep in range(1, args.max_epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        for k in range(0, len(perm), bs):
            batch = collate(tr, perm[k:k + bs].tolist(), pad_id)
            opt.zero_grad()
            loss = model.loss(*batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
        f1 = micro_f1(va_t, predict(model, va, va_s, labels, pad_id), types)
        if f1 > best_f1:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [{model_id} s{seed}] epoch {ep:02d} loss {loss.item():.4f} val micro-F1 {f1:.4f}"
              f"{'  *' if bad == 0 else ''}")
        if bad >= HP["patience"] and ep >= args.min_epochs:    # never stop before min_epochs
            break
    model.load_state_dict(best_state)
    pred = predict(model, te, te_s, labels, pad_id)
    test_f1 = micro_f1(te_t, pred, types)
    save_preds(model_id, seed, pred, {
        "config": {"checkpoint": path, "head": spec["head"], "weighted_loss": spec["weighted"],
                   "encoder_lr": lr, "head_lr": HP["head_lr"], "batch_size": bs,
                   "max_epochs": args.max_epochs, "patience": HP["patience"], "max_len": HP["max_len"],
                   "warmup": HP["warmup"], "weight_decay": HP["weight_decay"], "fine_tuned": True,
                   "fp16": use_amp, "selection": "val strict micro-F1",
                   "n_params": sum(p.numel() for p in model.parameters())},
        "val_micro_f1": best_f1, "best_epoch": best_ep, "test_words_truncated": n_trunc,
        "train_seconds": round(time.time() - t0, 1),
        "family": "Transformer" if spec["head"] == "linear" else "Hybrid"})
    print(f"[{model_id}] seed {seed}: best epoch {best_ep}, val {best_f1:.4f}, "
          f"test strict micro-F1 {test_f1:.4f}  (test words lost to truncation: {n_trunc})")
    del model, opt, best_state
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max_epochs", type=int, default=HP["max_epochs"])
    ap.add_argument("--min_epochs", type=int, default=HP["min_epochs"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--model_path", default=None, help="override checkpoint (single model only)")
    args = ap.parse_args()
    data, sp = get_all_splits()
    print("device:", dev, "| fp16:", use_amp)
    for m in args.models:
        for s in args.seeds:
            if already_done(m, s):
                print(f"[{m}] seed {s} already done - skipping")
                continue
            run(m, s, data, sp, args)


if __name__ == "__main__":
    main()
