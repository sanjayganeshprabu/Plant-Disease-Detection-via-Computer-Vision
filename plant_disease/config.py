"""Project-wide settings: paths, image size, and default hyperparameters.

Every value here can be overridden from the command line of the scripts
(`train.py`, `evaluate.py`, `predict.py`). The model directory used by the
web app can also be overridden with the ``PLANT_MODEL_DIR`` environment
variable, which is handy when deploying.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # where the Kaggle zip is extracted
TRAIN_DIR = DATA_DIR / "train"      # 38 class sub-folders
VALID_DIR = DATA_DIR / "valid"      # 38 class sub-folders
TEST_DIR = DATA_DIR / "test"        # a handful of unlabelled sample images

MODEL_DIR = Path(os.environ.get("PLANT_MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_FILENAME = "plant_disease_model.keras"
CLASS_NAMES_FILENAME = "class_names.json"
HISTORY_FILENAME = "training_history.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"   # plots, reports, metrics

# -------------------------------------------------------------- dataset
KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
ZIP_NAME = "new-plant-diseases-dataset.zip"

# ------------------------------------------------------------- training
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 3
SEED = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
