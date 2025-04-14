import re
import os
from pathlib import Path
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
        iter_germeval(file), n_sents=1, merge_subtokens=True, no_print=True
    )
    db = DocBin(docs=docs, store_user_data=True)
    doc_bins[bucket].merge(db)

msg.divider("Save splits to .spacy format")
# save splits to spacy doc format
for bucket, doc_bin in doc_bins.items():
    target_file = ner_dir / f"{bucket}.spacy"
    target_file.write_bytes(doc_bin.to_bytes())
