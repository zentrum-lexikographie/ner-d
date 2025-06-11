# ner-d
Pipline for training German [spaCy](https://spacy.io/) models for NER on a collection of gold standard corpora and a silver standard corpus (WikiNER).

This project trains a Named Entity Recognizer with labels `PER`, `ORG`, `LOC`, and `MISC`. It covers dataset download, data conversion and tagset normalization as well as training two models, one for GPU (`-dist`) and one for CPU (`-lg`) architectures.
After the training, the models are evaluated on the test split of the datasets as well as the test set of the CONLL03 dataset with the revision from 2006.

## Installation

For training on a GPU system:
```
pip install .[gpu]
```

For CPU-based training:

```
pip install .
```

## Training
```
GPU_ID=0 spacy project run all
```
Run the command without `GPU_ID=0` for CPU-based training.

### Step-by-step

```
spacy project download
spacy project preprocess
GPU_ID=0 spacy project train
```
## Publish model

Login with HuggingFace credential via `huggingface-cli` and run

```
python hf-publish.py
```

### Clean-Up
This removes the outputs of the `preprocess` and `train` step.
```
spacy project clean
```

## Datasets used for training
* D. Benikova, C. Biemann, M. Reznicek (2014). NoSta-D Named Entity Annotation for German: Guidelines and Dataset. Proceedings of LREC 2014, Reykjavik, Iceland.
* M. Schiersch, V. Mironova, M. Schmitt, P. Thomas, A. Gabryszak, L. Hennig (2018). A German Corpus for Fine-Grained Named Entity Recognition and Relation Extraction of Traffic and Industry Events. Proceedings of LREC 2018, Miyazaki, Japan.
* J. Zöllner, K. Sperfeld, C. Wick, R. Labahn (2021). Optimizing Small BERTs Trained for German NER. Information 2021, 12, 443.
* M. Ehrmann, M. Romanello, A. Flückiger, and S. Clematide (2020). Extended Overview of CLEF HIPE 2020: Named Entity Processing on Historical Newspapers in Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum, Thessaloniki, Greece, 2020, vol. 2696, p. 38. doi: 10.5281/zenodo.4117566.
* L. Hennig, P. T. Truong, A. Gabryszak (2021). Mobie: A German Dataset for Named Entity Recognition, Entity Linking and Relation Extraction in the Mobility Domain. arXiv preprint arXiv:2108.06955.
* A. Hamdi, E. Linhares Pontes, E. Boros, T. T. H. Nguyen, G. Hackl, J. G. Moreno, A- Doucet (2021). Multilingual Dataset for Named Entity Recognition, Entity Linking and Stance Detection in Historical Newspapers (V1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4573313
* J. Nothman, N. Ringland, W. Radford, T. Murphy, J. R. Curran (2013). Learning Multilingual Named Entity Recognition from Wikipedia. Artificial Intelligence, 194, 151-175.
* S. Schweter (2025). HisGermaNER (Revision 83571b3). doi: 10.57967/hf/5770, https://huggingface.co/datasets/stefan-it/HisGermaNER.
