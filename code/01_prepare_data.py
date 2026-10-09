#!/usr/bin/env python
"""
01_prepare_data.py  --  run ONCE, before any model.

  * reads PNER2.csv (columns: Sentence#, Word, Tag)
  * audits the corpus (sentence/token/entity counts, tag inventory, malformed BESO sequences,
    duplicate sentences)  -> numbers for the corrected Section 4.6 / Table 18
  * optional --fix_inside: relabels the tokens between B-X ... E-X as I-X (needed when entities of
    3+ tokens were annotated with O in the middle, reviewer point 4)
  * builds ONE sentence-level 80/10/10 split (train/val/test), stratified by entity type, with
    identical sentences kept in the same partition (no leakage)
  * writes work/data.json, work/splits.json, work/dataset_stats.csv, work/PNER_release.tsv

usage:  python 01_prepare_data.py --csv /content/PNER2.csv [--fix_inside]
"""
import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pner_common import WORK, extract_spans, split_tag


def norm_tag(t):
    t = str(t).strip()
    if t.upper() == "O":
        return "O"
    m = re.match(r"^([A-Za-z])[_\-](.+)$", t)
    if not m:
        raise ValueError(f"Unrecognised tag: {t!r}")
    p, ty = m.group(1).upper(), m.group(2).strip().upper()
    if p == "M":
        p = "I"
    if p not in "BIES":
        raise ValueError(f"Unrecognised tag prefix in {t!r}")
    return f"{p}-{ty}"


def audit(tags_list):
    """counts malformed sentences = sentences with a B/I/E tag that is not part of a valid span."""
    malformed = []
    for k, tags in enumerate(tags_list):
        covered = np.zeros(len(tags), dtype=bool)
        for s, e, _ in extract_spans(tags):
            covered[s:e + 1] = True
        if any(t != "O" and not covered[i] for i, t in enumerate(tags)):
            malformed.append(k)
    return malformed


def fix_inside(tags, max_gap):
    """B-X O O E-X  ->  B-X I-X I-X E-X   (only when the gap is all-O and <= max_gap)."""
    out, i, n, gaps = list(tags), 0, len(tags), []
    while i < n:
        p, ty = split_tag(out[i])
        if p == "B":
            j = i + 1
            while j < n and out[j] == "O":
                j += 1
            gap = j - i - 1
            if 0 < gap <= max_gap and j < n and out[j] == f"E-{ty}":
                for k in range(i + 1, j):
                    out[k] = f"I-{ty}"
                gaps.append(gap)
                i = j + 1
                continue
        i += 1
    return out, gaps


def count_gap_candidates(tags_list, max_gap):
    n = 0
    for tags in tags_list:
        _, g = fix_inside(tags, max_gap)
        n += len(g)
    return n


def diagnose(tags, max_gap):
    """why is a tag sequence still malformed?  returns a list of (token_index, problem)."""
    n = len(tags)
    cov = np.zeros(n, dtype=bool)
    for s, e, _ in extract_spans(tags):
        cov[s:e + 1] = True
    out = []
    for i, t in enumerate(tags):
        if t == "O" or cov[i]:
            continue
        p, ty = split_tag(t)
        if p == "B":
            j = i + 1
            while j < n and tags[j] == "O":
                j += 1
            if j >= n:
                out.append((i, "B never closed (sentence ends)"))
            else:
                pj, tj = split_tag(tags[j])
                if pj == "E" and tj == ty:
                    out.append((i, f"B ... E gap longer than {max_gap}"))
                elif pj == "E":
                    out.append((i, "B closed by E of a DIFFERENT type"))
                else:
                    out.append((i, "B never closed (next entity starts)"))
        elif p == "E":
            out.append((i, "E without matching B"))
        else:
            out.append((i, "I without matching B/E"))
    return out


def stratified_split(group_counts, group_sizes, rng, tries, frac=(0.8, 0.1, 0.1)):
    """random search over group permutations; keeps the per-type entity share of val/test
    as close as possible to 10 % (important for rare TIME / DATE entities)."""
    G, T = group_counts.shape
    total = group_counts.sum(axis=0).astype(float)
    N = group_sizes.sum()
    best, best_loss = None, 1e18
    for _ in range(tries):
        perm = rng.permutation(G)
        cs_sent = np.cumsum(group_sizes[perm])
        cs = np.cumsum(group_counts[perm], axis=0)
        i1 = int(np.searchsorted(cs_sent, frac[0] * N))
        i2 = int(np.searchsorted(cs_sent, (frac[0] + frac[1]) * N))
        if i1 < 1 or i2 <= i1 or i2 >= G:
            continue
        val = cs[i2 - 1] - cs[i1 - 1]
        test = total - cs[i2 - 1]
        tgt = frac[1] * total
        loss = (np.abs(val - tgt) / np.maximum(tgt, 1)).sum() + (np.abs(test - tgt) / np.maximum(tgt, 1)).sum()
        if loss < best_loss:
            best_loss, best = loss, (perm.copy(), i1, i2)
    perm, i1, i2 = best
    return perm[:i1], perm[i1:i2], perm[i2:], best_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split_seed", type=int, default=13)
    ap.add_argument("--tries", type=int, default=5000)
    ap.add_argument("--fix_inside", action="store_true",
                    help="relabel O-gaps between B-X and E-X as I-X (see audit output first)")
    ap.add_argument("--max_gap", type=int, default=8)
    ap.add_argument("--drop_malformed", action="store_true",
                    help="drop sentences that are still malformed after the optional fix")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- read
    df = pd.read_csv(args.csv, encoding="utf-8")
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    sid = cols.get("sentence#") or cols.get("sentence") or list(df.columns)[0]
    wcol = cols.get("word") or cols.get("token")
    tcol = cols.get("tag") or cols.get("label")
    df = df[[sid, wcol, tcol]].rename(columns={sid: "sid", wcol: "word", tcol: "tag"})
    df["sid"] = df["sid"].ffill()                       # ONLY the sentence id is forward-filled
    n_missing = int(df["word"].isna().sum() + df["tag"].isna().sum())
    df = df.dropna(subset=["word", "tag"]).copy()       # never forward-fill words or tags
    df["word"] = df["word"].astype(str).str.strip()
    n_empty = int((df["word"] == "").sum())
    df = df[df["word"] != ""]
    df["tag"] = df["tag"].map(norm_tag)

    sentences, tags, orig_ids = [], [], []
    for key, g in df.groupby("sid", sort=False):
        sentences.append(g["word"].tolist())
        tags.append(g["tag"].tolist())
        orig_ids.append(str(key))

    # ---------------------------------------------------------------- audit
    print("=" * 70, "\nCORPUS AUDIT (raw file)\n" + "=" * 70)
    print(f"rows with missing word/tag dropped : {n_missing}   empty words dropped: {n_empty}")
    n_tok = sum(len(s) for s in sentences)
    print(f"sentences                          : {len(sentences)}")
    print(f"tokens                             : {n_tok}")
    print(f"unique word types                  : {len(set(w for s in sentences for w in s))}")
    lens = np.array([len(s) for s in sentences])
    print(f"sentence length mean/95%/99%/max   : {lens.mean():.1f} / {np.percentile(lens, 95):.0f} / "
          f"{np.percentile(lens, 99):.0f} / {lens.max()}")
    prefix = Counter(split_tag(t)[0] for ts in tags for t in ts)
    print("tag prefixes                       :", dict(prefix))
    mal = audit(tags)
    gap_cand = count_gap_candidates(tags, args.max_gap)
    print(f"sentences with malformed BESO      : {len(mal)}")
    print(f"B-X O+ E-X gaps (O inside entity)  : {gap_cand}   (these are repaired by --fix_inside)")

    if args.fix_inside:
        all_gaps, new_tags = [], []
        for ts in tags:
            nt, g = fix_inside(ts, args.max_gap)
            new_tags.append(nt)
            all_gaps += g
        tags = new_tags
        print(f"\n--fix_inside: relabelled {len(all_gaps)} entities, gap-length histogram: "
              f"{dict(sorted(Counter(all_gaps).items()))}")
        mal = audit(tags)
        print(f"sentences still malformed after fix: {len(mal)}")

    if mal:
        diag = {k: diagnose(tags[k], args.max_gap) for k in mal}
        cats = Counter(pr for k in mal for _, pr in diag[k])
        print("\nRESIDUAL PROBLEMS (what is still wrong after the repair):")
        for pr, c in cats.most_common():
            print(f"   {c:5d}  {pr}")
        pd.DataFrame({"sentence_index": mal, "orig_id": [orig_ids[k] for k in mal],
                      "problem": [" | ".join(f"tok {i}: {pr}" for i, pr in diag[k]) for k in mal],
                      "words": [" ".join(sentences[k]) for k in mal],
                      "tags": [" ".join(tags[k]) for k in mal]}).to_csv(
            WORK / "malformed_sentences.csv", index=False, encoding="utf-8-sig")
        print(f"-> work/malformed_sentences.csv written ({len(mal)} rows, with a 'problem' column)")
        if args.drop_malformed:
            keep = [k for k in range(len(sentences)) if k not in set(mal)]
            sentences = [sentences[k] for k in keep]
            tags = [tags[k] for k in keep]
            orig_ids = [orig_ids[k] for k in keep]
            print(f"-> --drop_malformed: kept {len(keep)} sentences")

    labels = sorted({t for ts in tags for t in ts}, key=lambda x: (x != "O", x))
    types = sorted({split_tag(t)[1] for t in labels if t != "O"})
    print(f"\nlabel inventory ({len(labels)} labels): {labels}")
    print(f"entity types ({len(types)}): {types}")

    # ---------------------------------------------------------------- entity counts per sentence
    tix = {t: i for i, t in enumerate(types)}
    sent_counts = np.zeros((len(sentences), len(types)), dtype=int)
    for k, ts in enumerate(tags):
        for _, _, ty in extract_spans(ts):
            sent_counts[k, tix[ty]] += 1
    n_ent = int(sent_counts.sum())
    print(f"named entities (well-formed spans) : {n_ent}")
    print("per type                           :", {t: int(sent_counts[:, i].sum()) for t, i in tix.items()})

    # ---------------------------------------------------------------- duplicate-safe stratified split
    groups = OrderedDict()
    for k, s in enumerate(sentences):
        groups.setdefault(tuple(s), []).append(k)
    glist = list(groups.values())
    n_dup = len(sentences) - len(glist)
    print(f"duplicate sentences                : {n_dup}  (kept inside ONE partition)")
    gc = np.array([sent_counts[g].sum(axis=0) for g in glist])
    gs = np.array([len(g) for g in glist])
    rng = np.random.RandomState(args.split_seed)
    tr_g, va_g, te_g, loss = stratified_split(gc, gs, rng, args.tries)
    to_idx = lambda gi: sorted(k for g in gi for k in glist[g])
    split = {"train": to_idx(tr_g), "val": to_idx(va_g), "test": to_idx(te_g)}
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"]))
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(sentences)

    rows = []
    for name in ("train", "val", "test"):
        idx = split[name]
        r = {"split": name, "sentences": len(idx), "tokens": sum(len(sentences[i]) for i in idx),
             "entities": int(sent_counts[idx].sum())}
        for t in types:
            r[t] = int(sent_counts[idx, tix[t]].sum())
        rows.append(r)
    tot = {"split": "total", **{k: sum(r[k] for r in rows) for k in rows[0] if k != "split"}}
    stats = pd.DataFrame(rows + [tot])
    stats.to_csv(WORK / "dataset_stats.csv", index=False)
    print("\nSPLIT STATISTICS (put this in the paper)\n", stats.to_string(index=False))

    # split sanity: word overlap
    tr_words = set(w for i in split["train"] for w in sentences[i])
    te_tok = [w for i in split["test"] for w in sentences[i]]
    oov = np.mean([w not in tr_words for w in te_tok])
    print(f"\ntest tokens not seen in train (OOV rate): {oov:.3f}")

    # ---------------------------------------------------------------- write
    with open(WORK / "data.json", "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences, "tags": tags, "labels": labels, "types": types,
                   "orig_ids": orig_ids, "fix_inside": bool(args.fix_inside)}, f, ensure_ascii=False)
    with open(WORK / "splits.json", "w", encoding="utf-8") as f:
        json.dump({**{k: [int(i) for i in v] for k, v in split.items()},
                   "split_seed": args.split_seed, "unit": "sentence",
                   "stratified_by": "entity type", "duplicates_kept_together": True}, f)
    which = {i: n for n, idx in split.items() for i in idx}
    with open(WORK / "PNER_release.tsv", "w", encoding="utf-8") as f:
        f.write("sentence\tsplit\ttoken_id\tword\ttag\n")
        for k, (s, ts) in enumerate(zip(sentences, tags)):
            for j, (w, t) in enumerate(zip(s, ts)):
                f.write(f"{k}\t{which[k]}\t{j}\t{w}\t{t}\n")
    print("\nwritten: work/data.json  work/splits.json  work/dataset_stats.csv  work/PNER_release.tsv")


if __name__ == "__main__":
    main()
