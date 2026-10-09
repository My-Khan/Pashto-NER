#!/usr/bin/env python
"""
05_aggregate.py -- builds EVERYTHING reported in the paper from the saved prediction files.

Outputs (results/):
  main_table.csv / .tex            strict span-level micro P/R/F1 and macro-F1, mean +- SD over seeds,
                                   per-entity F1, plus Holm-adjusted significance versus the best model
  per_entity_f1.csv                per-type F1, mean +- SD over seeds           (supplementary table)
  per_entity_prf.csv               per-type precision, recall and F1 for every model (long format)
  hyperparameters.csv              one row per model: input, full configuration, selection criterion
  pairwise_pvalues_micro_f1.csv    Holm-adjusted paired-bootstrap p-values, all model pairs
  bootstrap_ci.csv                 95 % bootstrap CI of micro-/macro-F1 for every model
  environment.json                 software / hardware used
  fig1_model_comparison.pdf/png    micro-F1 (bar, 95 % CI) and macro-F1 (marker) of all models
  fig2_per_entity_f1.pdf/png       model x entity-type F1 heat-map
  fig3_confusion_best.pdf/png      span-level type confusion of the best model

Significance testing (reviewer points 6-7): paired bootstrap over TEST SENTENCES. All models are scored
on the same resampled sentences (statistic = mean over seeds of the per-seed F1), the p-value is the two-sided bootstrap
probability that the F1 difference has the opposite sign, and the Holm procedure corrects for the
number of comparisons. Seed variability is reported separately as mean +- SD.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pner_common import (MODEL_INFO, available_seeds, extract_spans, get_split, load_data, load_preds,
                         metrics_from_counts, sentence_counts)

RES = Path(os.environ.get("PNER_RESULTS", "results"))
FAM_COLOR = {"ML": "#4C72B0", "DL": "#55A868", "Transformer": "#C44E52", "Hybrid": "#8172B2"}
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def f1_from_sums(S):                                  # S: (..., T, 3)  [tp, fp, fn]
    tp, fp, fn = S[..., 0], S[..., 1], S[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def bootstrap(allC, support, B, seed=0, chunk=200):
    """paired bootstrap over test sentences. For every resample the statistic of a model is the
    MEAN OVER SEEDS of its per-seed F1, so the interval is centred on the number in the table."""
    ids = list(allC)
    n = next(iter(allC.values())).shape[1]
    rng = np.random.RandomState(seed)
    micro = {m: [] for m in ids}
    macro = {m: [] for m in ids}
    present = support > 0
    for start in range(0, B, chunk):
        cb = min(chunk, B - start)
        idx = rng.randint(0, n, size=(cb, n))                      # same resample for ALL models
        for m in ids:
            mi, ma = np.zeros(cb), np.zeros(cb)
            for C in allC[m]:                                      # C: (n_sent, T, 3) of one seed
                S = C[idx].sum(axis=1)                             # (cb, T, 3)
                mi += f1_from_sums(S.sum(axis=1))
                ma += f1_from_sums(S)[:, present].mean(axis=1)
            micro[m].append(mi / len(allC[m]))
            macro[m].append(ma / len(allC[m]))
    return ({m: np.concatenate(v) for m, v in micro.items()},
            {m: np.concatenate(v) for m, v in macro.items()})


def latex_table(df, path, caption, label):
    cols = list(df.columns)
    def esc(x):
        x = str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        return x.replace("±", r"$\pm$")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 3 + "c" * (len(cols) - 3) + "}", r"\hline",
             " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    last_family = None
    for _, r in df.iterrows():
        if last_family is not None and r["Family"] != last_family:
            lines.append(r"\hline")
        last_family = r["Family"]
        lines.append(" & ".join(esc(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def fmt(mean, sd, scale=100, nd=1):
    if np.isnan(sd):
        return f"{mean * scale:.{nd}f}"
    return f"{mean * scale:.{nd}f} ± {sd * scale:.{nd}f}"


def save_fig(fig, name):
    fig.savefig(RES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(RES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    data = load_data()
    types = data["types"]
    _, gold = get_split("test")

    per_seed, allC, meta = {}, {}, {}
    for m in MODEL_INFO:
        seeds = available_seeds(m)
        if not seeds:
            print(f"!! no predictions for {m} - skipped")
            continue
        runs, Cs = [], []
        for s in seeds:
            rec = load_preds(m, s)
            C = sentence_counts(gold, rec["preds"], types)
            runs.append(metrics_from_counts(C, types))
            Cs.append(C)
            meta.setdefault(m, rec["meta"])
        per_seed[m] = runs
        allC[m] = np.stack(Cs).astype(np.float32)
    models = list(per_seed)
    support = np.array([per_seed[models[0]][0]["per_type"][t]["support"] for t in types])
    print(f"{len(models)} models loaded; test set: {len(gold)} sentences, {int(support.sum())} entities")

    # ------------------------------------------------------------ bootstrap
    micro_b, macro_b = bootstrap(allC, support, args.boot)
    ci = {m: (np.percentile(micro_b[m], [2.5, 97.5]), np.percentile(macro_b[m], [2.5, 97.5])) for m in models}
    pd.DataFrame([{"model": MODEL_INFO[m][0], "micro_f1_ci_low": ci[m][0][0], "micro_f1_ci_high": ci[m][0][1],
                   "macro_f1_ci_low": ci[m][1][0], "macro_f1_ci_high": ci[m][1][1]} for m in models]
                 ).to_csv(RES / "bootstrap_ci.csv", index=False)

    pairs = list(itertools.combinations(models, 2))
    raw = []
    for a, b in pairs:
        d = micro_b[a] - micro_b[b]
        raw.append(min(1.0, 2 * (min(np.mean(d <= 0), np.mean(d >= 0)) * args.boot + 1) / (args.boot + 1)))
    adj = holm(raw)
    P = pd.DataFrame(np.nan, index=[MODEL_INFO[m][0] for m in models], columns=[MODEL_INFO[m][0] for m in models])
    for (a, b), p in zip(pairs, adj):
        P.loc[MODEL_INFO[a][0], MODEL_INFO[b][0]] = p
        P.loc[MODEL_INFO[b][0], MODEL_INFO[a][0]] = p
    P.to_csv(RES / "pairwise_pvalues_micro_f1.csv")

    # ------------------------------------------------------------ main table
    mean_micro = {m: np.mean([r["micro_f1"] for r in per_seed[m]]) for m in models}
    best = max(models, key=mean_micro.get)
    vs_best = [(m, raw[pairs.index((best, m)) if (best, m) in pairs else pairs.index((m, best))])
               for m in models if m != best]
    vs_best_adj = dict(zip([m for m, _ in vs_best], holm([p for _, p in vs_best])))   # family = comparisons with best

    rows, per_ent_rows = [], []
    for m in models:
        R = per_seed[m]
        sd = lambda key: np.std([r[key] for r in R], ddof=1) if len(R) > 1 else np.nan
        row = {"Model": MODEL_INFO[m][0], "Family": MODEL_INFO[m][1], "Input": MODEL_INFO[m][2], "Runs": len(R),
               "Micro-P": fmt(np.mean([r["micro_precision"] for r in R]), sd("micro_precision")),
               "Micro-R": fmt(np.mean([r["micro_recall"] for r in R]), sd("micro_recall")),
               "Micro-F1": fmt(mean_micro[m], sd("micro_f1")),
               "Macro-F1": fmt(np.mean([r["macro_f1"] for r in R]), sd("macro_f1")),
               "Micro-F1 95% CI": f"[{ci[m][0][0] * 100:.1f}, {ci[m][0][1] * 100:.1f}]"}
        if m == best:
            row["vs. best (Holm p)"] = "best"
        else:
            p = vs_best_adj[m]
            row["vs. best (Holm p)"] = f"{p:.3f}" + (" *" if p < 0.05 else " (n.s.)")
        pe = {}
        for t in types:
            vals = [r["per_type"][t]["f1"] for r in R]
            pe[t] = (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else np.nan)
            row[t] = f"{pe[t][0] * 100:.1f}"
        rows.append(row)
        per_ent_rows.append({"Model": MODEL_INFO[m][0], "Family": MODEL_INFO[m][1],
                             **{t: fmt(*pe[t]) for t in types}})
    T = pd.DataFrame(rows)
    T.to_csv(RES / "main_table.csv", index=False, encoding="utf-8-sig")
    latex_table(T.drop(columns=["Micro-P", "Micro-R"]), RES / "main_table.tex",
                "Strict span-level results on the common PNER test set (mean $\\pm$ SD over seeds; "
                "95\\% CI by paired bootstrap over test sentences).", "tab:main")
    pd.DataFrame(per_ent_rows).to_csv(RES / "per_entity_f1.csv", index=False, encoding="utf-8-sig")

    # per-type precision / recall / F1 (mean +- SD over seeds) for every model
    prf_rows = []
    for m in models:
        R = per_seed[m]
        for t in types:
            row = {"Model": MODEL_INFO[m][0], "Family": MODEL_INFO[m][1], "Entity type": t,
                   "Support": R[0]["per_type"][t]["support"]}
            for key, name in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
                vals = [r["per_type"][t][key] for r in R]
                row[name] = fmt(np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else np.nan)
            prf_rows.append(row)
    pd.DataFrame(prf_rows).to_csv(RES / "per_entity_prf.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ hyper-parameter / setup summary
    hp = []
    for m in models:
        mt = meta[m]
        cfg = "; ".join(f"{k}={v}" for k, v in mt.get("config", {}).items())
        sec = [load_preds(m, s)["meta"].get("train_seconds", np.nan) for s in available_seeds(m)]
        hp.append({"Model": MODEL_INFO[m][0], "Family": MODEL_INFO[m][1], "Input": MODEL_INFO[m][2],
                   "Setting": "fine-tuned" if MODEL_INFO[m][1] in ("Transformer",) or m.startswith("xlmr_")
                   else "trained from scratch", "Runs": len(sec), "Configuration": cfg,
                   "Val micro-F1": round(float(np.mean([load_preds(m, s)["meta"].get("val_micro_f1", np.nan)
                                                        for s in available_seeds(m)])), 4),
                   "Train s/run": round(float(np.nanmean(sec)), 1)})
    pd.DataFrame(hp).to_csv(RES / "hyperparameters.csv", index=False, encoding="utf-8-sig")
    (RES / "environment.json").write_text(json.dumps(meta[models[0]].get("env", {}), indent=2))

    # ------------------------------------------------------------ FIGURE 1 : model comparison
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ypos = np.arange(len(models))[::-1]
    for y, m in zip(ypos, models):
        mu = mean_micro[m] * 100
        lo, hi = ci[m][0] * 100
        ax.barh(y, mu, color=FAM_COLOR[MODEL_INFO[m][1]], alpha=0.85, height=0.7)
        ax.errorbar(mu, y, xerr=[[max(mu - lo, 0)], [max(hi - mu, 0)]], color="k", capsize=2, lw=0.8)
        ax.plot(np.mean([r["macro_f1"] for r in per_seed[m]]) * 100, y, "D", color="white",
                mec="k", ms=4.5, zorder=5)
        ax.text(hi + 0.6, y, f"{mu:.1f}", va="center", fontsize=7.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([MODEL_INFO[m][0] for m in models])
    ax.set_xlabel("Strict span-level F1 (%)   bar = micro-F1 with 95% bootstrap CI,  ◇ = macro-F1")
    lo_x = min(ci[m][0][0] for m in models) * 100
    ax.set_xlim(0, 100)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOR.values()],
              labels=["Classical ML", "Neural (scratch)", "Transformer", "Hybrid"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    save_fig(fig, "fig1_model_comparison")

    # ------------------------------------------------------------ FIGURE 2 : per-entity heat-map
    H = np.array([[np.mean([r["per_type"][t]["f1"] for r in per_seed[m]]) * 100 for t in types] for m in models])
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    im = ax.imshow(H, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t.title()}\n(n={s})" for t, s in zip(types, support)], fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_INFO[m][0] for m in models])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if H[i, j] > 60 else "black")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="F1 (%)")
    save_fig(fig, "fig2_per_entity_f1")

    # ------------------------------------------------------------ FIGURE 3 : confusion of best model
    s0 = available_seeds(best)[0]
    pred = load_preds(best, s0)["preds"]
    tix = {t: i for i, t in enumerate(types)}
    M = np.zeros((len(types), len(types) + 1))
    for g, p in zip(gold, pred):
        by_b = {(s, e): ty for s, e, ty in extract_spans(p)}
        for s, e, ty in extract_spans(g):
            M[tix[ty], tix[by_b[(s, e)]] if (s, e) in by_b else len(types)] += 1
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(types) + 1))
    ax.set_xticklabels([t.title() for t in types] + ["Missed /\nboundary err."], rotation=40, ha="right")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([t.title() for t in types])
    ax.set_xlabel("Predicted span (same boundaries)")
    ax.set_ylabel("Gold span")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.5 else "black")
    ax.set_title(f"{MODEL_INFO[best][0]} (seed {s0})", fontsize=9)
    save_fig(fig, "fig3_confusion_best")

    # ------------------------------------------------------------ console summary
    print("\n" + T[["Model", "Family", "Runs", "Micro-F1", "Macro-F1", "vs. best (Holm p)"]].to_string(index=False))
    print(f"\nbest model by mean micro-F1: {MODEL_INFO[best][0]}")
    print("written to", RES.resolve())


if __name__ == "__main__":
    main()
