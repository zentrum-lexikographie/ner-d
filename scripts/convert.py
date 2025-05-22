import bz2
import gzip
import io
import random
import re
import zipfile
from pathlib import Path

import jsonstream
from spacy.tokens import DocBin
from spacy.training.converters import conll_ner_to_docs, iob_to_docs
from wasabi import msg

random.seed(0)

project_dir = Path(__file__).parents[1].resolve()

assets_dir = project_dir / "assets"
corpus_dir = project_dir / "corpus"
corpus_dir.mkdir(parents=True, exist_ok=True)


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
    data = re.sub(r"(LOC|PER|OTH|ORG)(deriv|part)", r"O", data)
    data = re.sub(r"\bOTH\b", "MISC", data)
    return data


for bucket, file in germeval_splits.items():
    docs = conll_ner_to_docs(
        iter_germeval(file), n_sents=10, merge_subtokens=True, no_print=True
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")

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
    symbols_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U0001F300-\U0001F5FF]",
        re.UNICODE,
    )
    with gzip.open(file) as fh:
        data = jsonstream.loads(fh.read())
        for doc in data:
            text = doc["text"]["string"]
            # skip sentence if it contains emojis and symbols
            if symbols_pattern.search(text):
                continue
            for token in doc["tokens"]["array"]:
                start = token["span"]["start"]
                end = token["span"]["end"]
                tok = text[start:end]
                tok = re.sub(r"\s", "", tok)
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
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")


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
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")

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
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")

msg.divider("Preprocessing HisGermaNER")

hisGermaNER_splits = {
    "train": assets_dir / "HisGermaNER_v0_train.tsv",
    "dev": assets_dir / "HisGermaNER_v0_dev.tsv",
    "test": assets_dir / "HisGermaNER_v0_test.tsv",
}


def iter_his_german_ner(file):
    data = []
    with file.open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("TOKEN") or "DOCSTART" in line:
                continue
            elif not line.strip():
                data.append(line)
            else:
                line = line.split()
                data.append(" ".join(line[0:2]))
    data = "\n".join(data)
    return data


for bucket, file in hisGermaNER_splits.items():
    docs = conll_ner_to_docs(
        iter_his_german_ner(file),
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")

msg.divider("Preprocessing CLEF HIPE")

hipe_splits = {
    "train": assets_dir / "HIPE-data-v1.4-train-de.tsv",
    "dev": assets_dir / "HIPE-data-v1.4-dev-de.tsv",
    "test": assets_dir / "HIPE-data-v1.4-test-de.tsv",
}
HIPE_TO_CONLL = {
    "loc": "LOC",
    "pers": "PER",
    "org": "ORG",
    "prod": "MISC",
    "time": "O",
    "date": "O",
}


def iter_hipe(file, tag_mapping):
    with file.open() as fh:
        tokens = []
        for line in fh:
            if line.startswith("TOKEN") or line.startswith("#") or not line.strip():
                continue
            line_data = line.strip().split()
            tokens.append("\t".join(line_data[:2]))
            misc = set(line_data[9].split("|"))
            if "PySBDSegment" in misc:
                tokens.append("\n")
        data = "\n".join(tokens)
        tag_substitution_pattern = re.compile(
            "(%s)" % "|".join(map(re.escape, tag_mapping.keys()))
        )
        data = tag_substitution_pattern.sub(lambda x: tag_mapping[x.group()], data)
    return data


for bucket, file in hipe_splits.items():
    docs = conll_ner_to_docs(
        iter_hipe(file, HIPE_TO_CONLL),
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")
    doc_bins[bucket].merge(db)

msg.divider("Preprocessing MobIE dataset")
mobie_split = {
    "train": assets_dir / "train.conll2003",
    "dev": assets_dir / "dev.conll2003",
    "test": assets_dir / "test.conll2003",
}

MOBIE_TO_CONLL = {
    "date": "O",
    "disaster-type": "O",
    "distance": "O",
    "duration": "O",
    "location": "LOC",
    "location-city": "LOC",
    "location-route": "O",
    "location-stop": "LOC",
    "location-street": "LOC",
    "number": "O",
    "organization": "ORG",
    "organization-company": "ORG",
    "org-position": "O",
    "person": "PER",
    "time": "O",
    "trigger": "O",
    "event-cause": "O",
    "money": "O",
    "percent": "O",
    "set": "O",
}


def iter_mobie(file, tag_mapping):
    with file.open() as fh:
        tokens = []
        for line in fh:
            if line.startswith("-DOCSTART-"):
                continue
            if not line.strip():
                tokens.append(line)
            else:
                line_data = line.strip().split("\t")
                tokens.append("\t".join([line_data[0], line_data[-1]]))
        data = "\n".join(tokens)
        tag_substitution_pattern = re.compile(
            r"(%s)(?!-)" % "|".join(map(re.escape, tag_mapping.keys()))
        )
        data = tag_substitution_pattern.sub(lambda x: tag_mapping[x.group()], data)
        return data


for bucket, file in mobie_split.items():
    docs = conll_ner_to_docs(
        iter_mobie(file, MOBIE_TO_CONLL),
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")
    doc_bins[bucket].merge(db)


msg.divider("Preprocessing newseye dataset")
newseye_split = {
    "train": "NewsEye-GT-NER_EL_StD-v1/NewsEye-German/train.tsv",
    "dev": "NewsEye-GT-NER_EL_StD-v1/NewsEye-German/dev.tsv",
    "test": "NewsEye-GT-NER_EL_StD-v1/NewsEye-German/test.tsv",
}


def process_newseye(file):
    with zipfile.ZipFile("assets/newseye.zip") as z:
        with io.TextIOWrapper(z.open(file), encoding="utf-8") as f:
            tokens = []
            for line in f:
                if line.startswith("Token\tTag\t") or line.startswith("#"):
                    continue
                if not line.strip():
                    tokens.append(line)
                else:
                    line_data = line.strip().split("\t", 2)
                    token, tag, _ = line_data
                    if tag.endswith("HumanProd"):
                        tag = "O"
                    tokens.append("\t".join([token, tag]))
        return "\n".join(tokens)


for bucket, file in newseye_split.items():
    docs = conll_ner_to_docs(
        process_newseye(file),
        n_sents=10,
        merge_subtokens=True,
        no_print=True,
    )
    db = DocBin(docs=docs, store_user_data=True)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")
    doc_bins[bucket].merge(db)

msg.divider("Preprocessing WikiNER dataset")

with bz2.open(assets_dir / "wikiner.bz2", "rt", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]
random.shuffle(lines)
dev_size = test_size = int(0.1 * len(lines))
train_size = (len(lines) - dev_size) - test_size
train_lines = lines[:train_size]
dev_lines = lines[train_size : train_size + dev_size]
test_lines = lines[train_size + dev_size :]
wikiner_splits = {
    "train": iob_to_docs("\n".join(train_lines), n_sents=10, no_print=True),
    "dev": iob_to_docs("\n".join(dev_lines), n_sents=10, no_print=True),
    "test": iob_to_docs("\n".join(test_lines), n_sents=10, no_print=True),
}
for bucket, docs in wikiner_splits.items():
    db = DocBin(docs=docs, store_user_data=True)
    msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in {bucket}.")
    doc_bins[bucket].merge(db)

msg.divider("Save splits to .spacy format")
# save splits to spacy doc format
for bucket, doc_bin in doc_bins.items():
    msg.info(f"{len(doc_bin)} documents (~{len(doc_bin)*10} sentences) in {bucket}.")
    target_file = corpus_dir / f"{bucket}.spacy"
    target_file.write_bytes(doc_bin.to_bytes())

msg.divider("Process Conll2003 test set")
conll_splits = {
    "train": assets_dir / "conll03.train",
    "dev": assets_dir / "conll03.dev",
    "test": assets_dir / "conll03.test",
}
tags_file = assets_dir / "tags.deu"

with tags_file.open(encoding="iso-8859-1") as f_tags:
    for bucket, file in conll_splits.items():
        with file.open(encoding="iso-8859-1") as f:
            data = []
            for line in f:
                tag_line = next(f_tags)
                if line.startswith("-DOCSTART-"):
                    continue
                if not line.strip():
                    assert tag_line.strip() == ""
                    data.append(line)
                else:
                    token, _ = line.split(maxsplit=1)
                    _, _, _, tag = tag_line.split()
                    data.append("\t".join([token, tag]))
            else:
                data = "\n".join(data)
                docs = conll_ner_to_docs(
                    data, n_sents=10, merge_subtokens=True, no_print=True
                )
        db = DocBin(docs=docs, store_user_data=True)
        msg.info(f"{len(db)} documents (~{len(db)*10} sentences) in conll03 {bucket}.")
        target_file = corpus_dir / f"{bucket}_conll03.spacy"
        target_file.write_bytes(db.to_bytes())
