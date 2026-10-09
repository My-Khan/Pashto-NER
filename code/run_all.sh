#!/usr/bin/env bash
# Full pipeline. Every script skips runs whose prediction file already exists, so after a Colab
# disconnect just run the same line again.
set -e
export PNER_WORK=work PNER_RESULTS=results

# 0) (optional) dry run of the whole pipeline on synthetic data - takes a few minutes
#    python make_toy_data.py toy.csv && PNER_WORK=work_toy python 01_prepare_data.py --csv toy.csv

# 1) audit + ONE common split.  Read the audit output before continuing!
python 01_prepare_data.py --csv /content/PNER2.csv            # add --fix_inside if the audit reports O-gaps

# 2) models (each family can be run in a separate session)
python 02_ml_models.py --xgb_device cuda
python 03_dl_models.py
python 04_transformers.py --models mbert distilmbert xlmr xlmr_weighted xlmr_crf xlmr_bilstm_crf
python 04_transformers.py --models xlmr_large xlmr_large_crf

# 3) tables, significance tests, figures
python 05_aggregate.py --boot 10000
