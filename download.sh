#!/bin/bash

# download datasets

# GermEval
gdown https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J --folder -O assets
rm assets/*pdf assets/CHANGELOG assets/NER-de-test-unlabeled.tsv

# SmartData
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/test.json.gz" -O assets/smartdata_test.json.gz
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/train.json.gz" -O assets/smartdata_train.json.gz
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/dev.json.gz" -O assets/smartdata_dev.json.gz

# Sturm Edition
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Sturm/test_sturm.conll" -O assets/test_sturm.conll
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Sturm/train_sturm.conll" -O assets/train_sturm.conll
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Sturm/dev_sturm.conll" -O assets/dev_sturm.conll

# Arendt Edition
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Arendt/test_arendt.conll" -O assets/test_arendt.conll
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Arendt/train_arendt.conll" -O assets/train_arendt.conll
wget -c "https://raw.githubusercontent.com/NEISSproject/NERDatasets/refs/heads/main/Arendt/dev_arendt.conll" -O assets/dev_arendt.conll

# hisGermanNER
wget -c "https://huggingface.co/datasets/stefan-it/HisGermaNER/resolve/main/splits/HisGermaNER_v0_dev.tsv" -O assets/HisGermaNER_v0_dev.tsv
wget -c "https://huggingface.co/datasets/stefan-it/HisGermaNER/resolve/main/splits/HisGermaNER_v0_test.tsv" -O assets/HisGermaNER_v0_test.tsv
wget -c "https://huggingface.co/datasets/stefan-it/HisGermaNER/resolve/main/splits/HisGermaNER_v0_train.tsv" -O assets/HisGermaNER_v0_train.tsv
