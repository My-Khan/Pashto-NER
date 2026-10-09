# PNER – unified re-experiment package

One data split, one evaluator, one prediction format for all 22 models.

## Run order (Colab, T4 GPU)

```bash
pip install -r requirements.txt
bash run_all.sh            # or run the steps one by one; every script resumes after a disconnect
```

| Step | Script | Output |
|---|---|---|
| 1 | `01_prepare_data.py --csv PNER2.csv` | `work/data.json`, `work/splits.json`, `work/dataset_stats.csv`, `work/PNER_release.tsv`, audit printout |
| 2 | `02_ml_models.py` | CRF, HMM, SVM, LR, RF, XGBoost, NB → `work/preds/` |
| 3 | `03_dl_models.py` | RNN, LSTM, Bi-LSTM, GRU, Bi-GRU, Bi-LSTM+CRF, CNN+Bi-GRU+Attn |
| 4 | `04_transformers.py` | mBERT, Distil-mBERT, XLM-R, XLM-R weighted, XLM-R large, +CRF, +Bi-LSTM+CRF, large+CRF |
| 5 | `05_aggregate.py` | `results/`: tables (csv + tex), significance tests, 3 figures, hyper-parameter table |

Total 22 models = 7 ML + 5 DL + 5 transformer + 5 hybrid (so "twenty-two" in the paper becomes true and
countable; Naive Bayes and HMM are both included).

## What changed relative to the old notebooks (reviewer point -> fix)

| Reviewer point | Fix in this package |
|---|---|
| R2-1 different test sets | every model reads `splits.json`; the test set is identical (same sentences, same order) |
| R2-2 / R4 split contradiction | sentence-level 80/10/10, stratified by entity type, duplicates kept together; validation used for tuning / early stopping, test predicted once; `dataset_stats.csv` gives sentences/tokens/entities per split; `PNER_release.tsv` gives the exact partition |
| R2-3 corpus statistics | the audit printout gives the true counts (sentences, tokens, unique words, entities, label inventory) – copy them into Table 18 |
| R2-4 BESO / long entities | audit counts malformed sequences and `B-X O.. E-X` gaps; `--fix_inside` converts them to `B-X I-X.. E-X` (BIOES). Describe the final scheme in the paper |
| R2-5 evaluation | `pner_common.py`: strict span-level P/R/F1 (exact boundary + type, O never counted). Verified identical to `seqeval` strict/IOBES on random predictions. Transformers predict on the first sub-token, so evaluation is word-level for all models |
| R2-6 / R2-7 statistics | ANOVA / t-test / Tukey on entity F1 are removed. Replaced by (a) mean ± SD over 5 seeds and (b) paired bootstrap over test sentences with Holm correction |
| R2-8 reproducibility | all hyper-parameters, seeds, selection criterion, hardware and library versions are written automatically to `results/hyperparameters.csv` and `results/environment.json` (one table replaces pages of textbook equations) |
| R2-9 frozen vs fine-tuned | all transformers are fully fine-tuned; this is recorded in the configuration |
| R2-10 model count | `MODEL_INFO` in `pner_common.py` is the single list of models |
| R4 Table 33 N/A, zero support | disappear automatically: every class has the same support for every model |

## Figures kept (everything else goes to the supplement or is deleted)

1. `fig1_model_comparison` – micro-F1 with 95 % CI (bars) and macro-F1 (markers) for all 22 models
2. `fig2_per_entity_f1` – model x entity-type heat-map
3. `fig3_confusion_best` – span-level confusion of the best model

Per-entity tables are in `per_entity_f1.csv` (supplementary). Remove the other bar charts, per-model confusion
matrices, box-plots and statistical plots.

## Things the code cannot do for you

* Source-wise splitting: the old text says sentences from the same news source were kept together. The CSV has
  no document id, so the split only keeps *identical sentences* together. Delete the source claim, or add a
  document-id column and group on it in `01_prepare_data.py`.
* Release the dataset, `splits.json`/`PNER_release.tsv`, annotation guidelines and this code in a public repository
  (Zenodo / GitHub) and state the BBC / VOA redistribution terms.
* Narrow "domain-independent" to "news-domain" unless you add an out-of-domain test set.
* Inter-annotator agreement: state at which level kappa was computed (token or entity).

## Smoke test before the long run

```bash
python make_toy_data.py toy.csv
export PNER_WORK=work_toy PNER_RESULTS=results_toy
python 01_prepare_data.py --csv toy.csv
python 02_ml_models.py --seeds 42 --rf_trees 20
python 03_dl_models.py --models bilstm bilstm_crf cnn_bigru_attn --seeds 42 --max_epochs 2
python 04_transformers.py --models xlmr xlmr_crf --seeds 42 --max_epochs 1
python 05_aggregate.py --boot 200
```
