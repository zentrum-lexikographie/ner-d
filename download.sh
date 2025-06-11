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

# CLEF HIPE
wget -c "https://raw.githubusercontent.com/impresso/CLEF-HIPE-2020/refs/heads/master/data/v1.4/de/HIPE-data-v1.4-dev-de.tsv" -O assets/HIPE-data-v1.4-dev-de.tsv
wget -c "https://raw.githubusercontent.com/impresso/CLEF-HIPE-2020/refs/heads/master/data/v1.4/de/HIPE-data-v1.4-train-de.tsv" -O assets/HIPE-data-v1.4-train-de.tsv
wget -c "https://raw.githubusercontent.com/impresso/CLEF-HIPE-2020/refs/heads/master/data/v1.4/de/HIPE-data-v1.4-test-de.tsv" -O assets/HIPE-data-v1.4-test-de.tsv

# mobie
wget -c "https://github.com/DFKI-NLP/MobIE/raw/refs/heads/master/v1_20210811/ner_conll03_formatted.zip" -O assets/mobie.zip
unzip assets/mobie.zip -d assets/

# newseye
wget -c "https://zenodo.org/records/4573313/files/NewsEye-GT-NER_EL_StD-v1.zip?download=1" -O assets/newseye.zip

# conll2003
wget -c "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb" -O assets/conll03.test
# testa corresponds to dev, cf. https://www.clips.uantwerpen.be/conll2003/ner/
wget -c "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa" -O assets/conll03.dev
wget -c "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train" -O assets/conll03.train
# updated annotations
wget -c "https://www.clips.uantwerpen.be/conll2003/ner.tgz" -O assets/ner.tgz
tar -zxv -C assets -f assets/ner.tgz ner/etc.2006/tags.deu  --strip-components=2

# wikiner
wget -c "https://github.com/dice-group/FOX/raw/refs/tags/v2.3.0/input/Wikiner/aij-wikiner-de-wp3.bz2" -O assets/wikiner.bz2
