import gzip
import re
import os
from pathlib import Path

import jsonstream
from spacy.tokens import DocBin
from spacy.training.converters import conll_ner_to_docs
from wasabi import msg

project_dir = Path(__file__).parent.resolve()

assets_dir = project_dir / "assets"
corpus_dir = project_dir / "corpus"
ner_dir = corpus_dir / "ner"
ner_dir.mkdir(parents=True, exist_ok=True)


doc_bins = {
    "train": DocBin(store_user_data=True),
    "dev": DocBin(store_user_data=True),
    "test": DocBin(store_user_data=True),
}

msg.divider("Preprocessing GermEval")

germeval_splits = {
    "train": assets_dir / "NER-de-train.tsv",
    "dev": assets_dir / "NER-de-dev.tsv",
    "test": assets_dir / "NER-de-test.tsv",
}


def iter_germeval(file):
    data = []
    with file.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            elif not line.strip():
                data.append(line)
            else:
                line = line.split()
                data.append(" ".join(line[1:3]))
    data = "\n".join(data)
    # remove *part from tags
    data = re.sub(r"(LOC|PER|OTH|ORG)(deriv|part)", r"\1", data)
    data = re.sub(r"\bOTH\b", "MISC", data)
    return data


for bucket, file in germeval_splits.items():
    docs = conll_ner_to_docs(
        iter_germeval(file), n_sents=32, merge_subtokens=True, no_print=True
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)

msg.divider("Preprocessing SmartData")

smartdata_splits = {
    "train": assets_dir / "smartdata_train.json.gz",
    "dev": assets_dir / "smartdata_dev.json.gz",
    "test": assets_dir / "smartdata_test.json.gz",
}

SMARTDATA_TO_CONLL = {
    "DATE": "O",
    "DISASTER_TYPE": "O",
    "DISTANCE": "O",
    "DURATION": "O",
    "LOCATION": "LOC",
    "LOCATION_CITY": "LOC",
    "LOCATION_ROUTE": "O",
    "LOCATION_STOP": "LOC",
    "LOCATION_STREET": "LOC",
    "NUMBER": "O",
    "ORGANIZATION": "ORG",
    "ORGANIZATION_COMPANY": "ORG",
    "ORG_POSITION": "O",
    "PERSON": "PER",
    "TIME": "O",
    "TRIGGER": "O",
}


def iter_smartdata(file, tag_mapping):
    tokens = []
    with gzip.open(file) as fh:
        data = jsonstream.loads(fh.read())
        for doc in data:
            text = doc["text"]["string"]
            for token in doc["tokens"]["array"]:
                start = token["span"]["start"]
                end = token["span"]["end"]
                tok = text[start:end]
                tags = token["ner"]["string"].split("-", 1)
                if len(tags) == 2:
                    iob, ent = tags
                    ent = tag_mapping[ent]
                    tag = "-".join([iob, ent]) if ent != "O" else ent
                else:
                    tag = tags[0]
                tokens.append("\t".join([tok, tag]))
            tokens.append("\n")
    return "\n".join(tokens)


for bucket, file in smartdata_splits.items():
    docs = conll_ner_to_docs(
        iter_smartdata(file, SMARTDATA_TO_CONLL),
        n_sents=32,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)


def iter_neiss_data(file, tag_mapping):
    data = []
    with file.open() as f:
        data = f.read()
        tag_substitution_pattern = re.compile(
            "(%s)" % "|".join(map(re.escape, tag_mapping.keys()))
        )
        return tag_substitution_pattern.sub(lambda x: tag_mapping[x.group()], data)


msg.divider("Preprocessing Sturm Edition")

sturm_ed_splits = {
    "train": assets_dir / "train_sturm.conll",
    "dev": assets_dir / "dev_sturm.conll",
    "test": assets_dir / "test_sturm.conll",
}
STURM_TO_CONLL = {"pers": "PER", "place": "LOC", "B-date": "O", "I-date": "O"}


for bucket, file in sturm_ed_splits.items():
    docs = conll_ner_to_docs(
        iter_neiss_data(file, STURM_TO_CONLL),
        n_sents=32,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)

msg.divider("Preprocessing Arendt Edition")

arendt_splits = {
    "train": assets_dir / "train_arendt.conll",
    "dev": assets_dir / "dev_arendt.conll",
    "test": assets_dir / "test_arendt.conll",
}
ARENDT_TO_CONLL = {
    "I-date": "O",
    "B-date": "O",
    "person": "PER",
    "ethnicity": "MISC",
    "organization": "ORG",
    "place": "LOC",
    "event": "MISC",
    "I-language": "O",
    "B-language": "O",
}
for bucket, file in arendt_splits.items():
    docs = conll_ner_to_docs(
        iter_neiss_data(file, ARENDT_TO_CONLL),
        n_sents=32,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)

msg.divider("Save splits to .spacy format")
# save splits to spacy doc format
for bucket, doc_bin in doc_bins.items():
    target_file = ner_dir / f"{bucket}.spacy"
    target_file.write_bytes(doc_bin.to_bytes())
