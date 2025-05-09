import os
from pathlib import Path

import spacy
from spacy.cli.download import download
from spacy.cli.evaluate import evaluate
from spacy.cli.package import package
from spacy.cli.train import train
from wasabi import msg

version = (Path(__file__).parents[1] / "VERSION").read_text().strip()
gpu_id = int(os.environ.get("GPU_ID", "-1"))

project_dir = Path(__file__).parents[1].resolve()

corpus_dir = project_dir / "corpus"
configs_dir = project_dir / "configs"
training_dir = project_dir / "training"
metrics_dir = project_dir / "metrics"
packages_dir = project_dir / "packages"

model_types = ["lg"] if gpu_id < 0 else ["lg", "dist"]

try:
    spacy.load("de_core_news_lg")
except IOError:
    download("de_core_news_lg")

for model_type in model_types:
    prefix = f"{model_type}-{version}"
    msg.divider(f"Pipeline: {prefix}")
    cfg_file_name = f"ner-d-{prefix}.cfg"
    cfg_path = configs_dir / cfg_file_name
    training_path = training_dir / prefix
    train(cfg_path, training_path, use_gpu=gpu_id)
    model_path = training_path / "model-best"
    evaluate(
        str(model_path),
        corpus_dir / "test.spacy",
        metrics_dir / f"{prefix}.json",
        silent=False,
        use_gpu=gpu_id,
    )
    evaluate(
        str(model_path),
        corpus_dir / "test_conll03.spacy",
        metrics_dir / f"conll03-{prefix}.json",
        silent=False,
        use_gpu=gpu_id,
    )
    package(
        model_path,
        packages_dir,
        name=f"ner-d_{model_type}",
        version=version,
        create_sdist=False,
        force=True,
        silent=False,
    )
