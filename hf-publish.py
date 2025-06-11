from pathlib import Path

from spacy_huggingface_hub import push

project_dir = Path(__file__).parent
packages_dir = project_dir / "packages"

organization = "zentrum-lexikographie"
version = (project_dir / "VERSION").read_text().strip()

for whl in packages_dir.rglob(f"*{version}*.whl"):
    push(whl.as_posix(), namespace=organization)
