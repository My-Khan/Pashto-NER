# PNER: an audited Pashto named entity recognition corpus and a 22-model benchmark

This repository accompanies the paper *Bridging the Resource Gap: A Multi-Paradigm Benchmark for Named Entity Recognition in Low-Resource Languages* ([journal, year, DOI to be added]).

The release contains the annotated Pashto sentences.

## Summary

| | |
|---|---|
| Language / domain | Pashto, news text (BBC Pashto and VOA Pashto) |
| Sentences / tokens / entities | 3,717 / 98,867 / 8,763 |
| Entity types (7) | PERSON, LOCATION, ORGANIZATION, DESIGNATION, NUMBER, DATE, TIME |
| Tagging scheme | BIOES, 29 labels (flat annotation; nested entities are not annotated) |
| Split (train / validation / test, sentences) | 2973 / 372 / 372 (sentence level, stratified by entity type, identical sentences kept together) |
| Annotators / agreement | 3 / Fleiss' kappa = 0.78 |
| Models | 22 (7 classical, 5 recurrent, 5 fine-tuned transformers, 5 hybrids) |
| Prediction files | 56 (3 seeds for stochastic models, 1 run for deterministic ones) |

## Repository layout

```
data/          corpus, fixed split, statistics, list of excluded sentences
predictions/   test-set predictions of every model and seed (one JSON file per run)
results/       tables, figures, significance tests and hyperparameters used in the paper
code/          scripts used for the experiments (01 data audit and split ... 05 aggregation)
```

## Data

* `data/PNER_release.tsv` : one row per token (`sentence`, `split`, `token_id`, `word`, `tag`). Tags follow BIOES, e.g. `B-ORGANIZATION`, `I-ORGANIZATION`, `E-ORGANIZATION`, `S-PERSON`, `O`.
* `data/splits.json` : sentence indices of the train, validation and test sets.
* `data/excluded_sentences.csv` : the 235 sentences with irreparable annotation errors that were removed, with a description of the problem. Correcting them would restore the full corpus.
* **Audit and repair.** The raw file had 3,952 sentences. In 1,199 entities of three or more tokens the interior tokens were labelled `O`; they were relabelled `I-<TYPE>`.
* **Known limitations.** News domain only; no document identifiers (sentences of one article may fall into different partitions); flat annotation; only 14 TIME entities in the test set.

## Reproducing the tables and figures

```bash
pip install -r code/requirements.txt
mkdir work && cp data/data.json data/splits.json work/ && cp -r predictions work/preds
PNER_WORK=work PNER_RESULTS=reproduced python code/05_aggregate.py --boot 10000
```

To retrain the models, run `code/01_prepare_data.py` on the raw annotations (see `code/README.md`) and then scripts 02 to 04 (a GPU is needed for the transformers).

## Licence

* Code: [choose, e.g. MIT]
* Annotations and corpus: [choose, e.g. CC BY 4.0 or CC BY-NC 4.0, depending on the BBC / VOA terms]
* Source texts: [state the terms of BBC Pashto and VOA Pashto and what is redistributed]

## Citation

Please cite the paper ([BibTeX to be added]) and the archived release ([Zenodo DOI to be added]).
