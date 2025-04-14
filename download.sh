#!/bin/bash

# download datasets

# GermEval
gdown https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J --folder -O assets
rm assets/*pdf assets/CHANGELOG assets/NER-de-test-unlabeled.tsv

# SmartData
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/test.json.gz" -O assets/smartdata_test.json.gz
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/train.json.gz" -O assets/smartdata_train.json.gz
wget -c "https://github.com/DFKI-NLP/smartdata-corpus/raw/master/v3_20200302/dev.json.gz" -O assets/smartdata_dev.json.gz
