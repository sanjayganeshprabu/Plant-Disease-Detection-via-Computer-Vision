"""Everything to do with getting the dataset onto disk and into TensorFlow."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from . import config


# ----------------------------------------------------------------------
# Download / extract / organise
# ----------------------------------------------------------------------
def download_from_kaggle(dest_dir: Path = config.DATA_DIR,
                         dataset: str = config.KAGGLE_DATASET) -> Path:
    """Download the dataset zip with the Kaggle API and return its path.

    Credentials are read by the Kaggle package from ``~/.kaggle/kaggle.json``
    or from the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The 'kaggle' package is missing. Run: pip install kaggle") from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / config.ZIP_NAME
    if zip_path.exists():
        print(f"[data] Zip already present, skipping download: {zip_path}")
        return zip_path

    api = KaggleApi()
    api.authenticate()
    print(f"[data] Downloading '{dataset}' (~2.7 GB) ...")
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=False, quiet=False)
    return zip_path


def extract_zip(zip_path: Path, extract_to: Path = config.RAW_DIR) -> Path:
    """Extract the Kaggle zip (skipped if already extracted)."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[data] Already extracted, skipping: {extract_to}")
        return extract_to
    print(f"[data] Extracting {zip_path} -> {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def _find_split_dir(root: Path, name: str, want_subdirs: bool) -> Path:
    """Find a folder called ``name`` anywhere under ``root``.

    ``want_subdirs=True`` looks for a folder of class sub-folders (train/valid);
    ``want_subdirs=False`` looks for a folder that directly holds image files
    (the Kaggle zip's flat ``test/test`` folder). This replaces the fragile
    ``os.listdir(...)[1]`` indexing used in the original notebook.
    """
    candidates = sorted(p for p in root.rglob(name) if p.is_dir())
    for cand in candidates:
        children = list(cand.iterdir())
        if want_subdirs and any(c.is_dir() for c in children):
            return cand
        if not want_subdirs and any(
            c.is_file() and c.suffix.lower() in config.IMAGE_EXTENSIONS for c in children
        ):
            return cand
    raise FileNotFoundError(f"Could not find a '{name}' folder inside {root}")


def organise_dataset(raw_dir: Path = config.RAW_DIR,
                     train_dir: Path = config.TRAIN_DIR,
                     valid_dir: Path = config.VALID_DIR,
                     test_dir: Path = config.TEST_DIR) -> None:
    """Move train/valid/test out of the nested Kaggle layout into ``data/``."""
    moves = [
        (_find_split_dir(raw_dir, "train", want_subdirs=True), train_dir),
        (_find_split_dir(raw_dir, "valid", want_subdirs=True), valid_dir),
    ]
    try:
        moves.append((_find_split_dir(raw_dir, "test", want_subdirs=False), test_dir))
    except FileNotFoundError:
        print("[data] No sample test folder found (optional) - skipping.")

    for src, dst in moves:
        if dst.exists():
            print(f"[data] {dst} already exists, leaving it alone.")
            continue
        shutil.move(str(src), str(dst))
        print(f"[data] {src}  ->  {dst}")


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)


# ----------------------------------------------------------------------
# TensorFlow datasets
# ----------------------------------------------------------------------
def make_dataset(directory: Path, *, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
                 shuffle: bool, seed: int = config.SEED):
    """Build a batched ``tf.data.Dataset`` of (image, one-hot label).

    Images come out as float32 in the 0-255 range; rescaling happens *inside*
    the model, so training and inference can never disagree on preprocessing.
    """
    import tensorflow as tf

    if not Path(directory).is_dir():
        raise FileNotFoundError(
            f"{directory} not found. Run `python download_data.py` first "
            "(or pass the correct --data-dir)."
        )
    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="categorical",
        image_size=tuple(img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    class_names = list(ds.class_names)
    return ds.prefetch(tf.data.AUTOTUNE), class_names


# ----------------------------------------------------------------------
# Class-name persistence (saved next to the model for inference)
# ----------------------------------------------------------------------
def save_class_names(class_names: list[str], model_dir: Path = config.MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / config.CLASS_NAMES_FILENAME
    path.write_text(json.dumps(class_names, indent=2))
    return path


def load_class_names(model_dir: Path = config.MODEL_DIR) -> list[str]:
    path = Path(model_dir) / config.CLASS_NAMES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train a model first (python train.py).")
    return json.loads(path.read_text())
