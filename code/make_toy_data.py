#!/usr/bin/env python
"""
make_toy_data.py -- creates a small synthetic file in the same format as PNER2.csv
(Sentence#, Word, Tag; BESO tags with '_' separator, sentence id only on the first row).
Use it ONLY to smoke-test the pipeline (python make_toy_data.py toy.csv); never for results.
"""
import random
import sys

random.seed(0)
PER = ["احمد", "کریم", "مینه", "رحمان", "نور", "گل"]
LOC = ["کابل", "پېښور", "کندهار", "بنو", "اسلام آباد", "خیبر پښتونخوا"]
ORG = ["د ملګرو ملتونو سازمان", "پوهنتون", "بي بي سي", "حکومت"]
DES = ["وزیر", "رییس", "مشر", "استاد"]
NUM = ["۱۰", "۲۰۰", "پنځه", "شل"]
DAT = ["۲۰۲۴ کال", "سه شنبه", "پرون"]
TIM = ["۱۰ بجې", "سهار"]
FILL = ["د", "په", "کې", "ویلي", "چې", "دی", "او", "هغه", "نن", "راغی", "ښار", "خبرې", "کړې"]
TYPES = [("PERSON", PER), ("LOCATION", LOC), ("ORGANIZATION", ORG), ("DESIGNATION", DES),
         ("NUMBER", NUM), ("DATE", DAT), ("TIME", TIM)]
WEIGHTS = [4, 5, 3, 2, 3, 1, 1]


def ent(ty, surface):
    toks = surface.split()
    if len(toks) == 1:
        return [(toks[0], f"S_{ty}")]
    out = [(toks[0], f"B_{ty}")]
    out += [(t, f"I_{ty}") for t in toks[1:-1]]      # genuine middle tokens
    out.append((toks[-1], f"E_{ty}"))
    return out


def main(path, n=600):
    rows = []
    for k in range(n):
        sent = []
        for _ in range(random.randint(1, 3)):
            sent += [(random.choice(FILL), "O") for _ in range(random.randint(1, 3))]
            ty, pool = random.choices(TYPES, weights=WEIGHTS)[0]
            sent += ent(ty, random.choice(pool))
        sent += [(random.choice(FILL), "O")]
        for j, (w, t) in enumerate(sent):
            rows.append((f"Sentence: {k + 1}" if j == 0 else "", w, t))
    with open(path, "w", encoding="utf-8") as f:
        f.write("Sentence#,Word,Tag\n")
        for s, w, t in rows:
            f.write(f"{s},{w},{t}\n")
    print(f"wrote {path}: {n} sentences, {len(rows)} tokens")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy.csv")
