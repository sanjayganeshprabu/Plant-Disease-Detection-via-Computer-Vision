"""Load a trained model and predict plant disease / health for leaf images.

Used by ``predict.py`` (command line) and ``app.py`` (Streamlit web app).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from PIL import Image

from . import config

ImageInput = Union[str, Path, Image.Image, np.ndarray]


# ----------------------------------------------------------------------
# Label helpers (pure Python - no TensorFlow needed)
# ----------------------------------------------------------------------
def _tidy(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def parse_label(class_name: str) -> tuple[str, str, bool]:
    """Split a folder-style class name into (plant, condition, is_healthy).

    >>> parse_label("Corn_(maize)___Common_rust_")
    ('Corn (maize)', 'Common rust', False)
    >>> parse_label("Tomato___healthy")
    ('Tomato', 'Healthy', True)
    """
    plant, _, condition = class_name.partition("___")
    is_healthy = "healthy" in condition.lower()
    condition = "Healthy" if is_healthy else _tidy(condition)
    return _tidy(plant), condition, is_healthy


@dataclass
class Prediction:
    class_name: str
    plant: str
    condition: str
    is_healthy: bool
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.is_healthy:
            return f"🌱 Healthy {self.plant} leaf ({self.confidence:.1%})"
        return f"⚠️ Unhealthy {self.plant} leaf - Disease: {self.condition} ({self.confidence:.1%})"


# ----------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------
class PlantDiseasePredictor:
    """Wraps a saved Keras model + its class names.

    Example::

        predictor = PlantDiseasePredictor.from_dir("models")
        print(predictor.predict("leaf.jpg").summary)
    """

    def __init__(self, model, class_names: list[str]):
        self.model = model
        self.class_names = class_names
        self.img_size = tuple(model.input_shape[1:3])  # (height, width)
        n_out = model.output_shape[-1]
        if n_out != len(class_names):
            raise ValueError(f"Model has {n_out} outputs but {len(class_names)} class names were given.")

    @classmethod
    def from_dir(cls, model_dir: Union[str, Path] = config.MODEL_DIR,
                 model_filename: str = config.MODEL_FILENAME) -> "PlantDiseasePredictor":
        from tensorflow import keras

        from .data import load_class_names

        model_path = Path(model_dir) / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found - train a model first (python train.py).")
        model = keras.models.load_model(str(model_path), compile=False)
        return cls(model, load_class_names(Path(model_dir)))

    # -- preprocessing --------------------------------------------------
    def preprocess(self, image: ImageInput) -> np.ndarray:
        """Return a float32 array of shape (H, W, 3) with raw 0-255 values.

        Uses the same bilinear ``tf.image.resize`` as training, and no manual
        /255 - the model rescales internally.
        """
        import tensorflow as tf

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        arr = np.asarray(image)
        if arr.ndim == 2:  # greyscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3]  # drop alpha if present
        resized = tf.image.resize(arr, self.img_size, method="bilinear")
        return resized.numpy().astype("float32")

    # -- prediction -----------------------------------------------------
    def _to_prediction(self, probs: np.ndarray, top_k: int) -> Prediction:
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        name = self.class_names[best]
        plant, condition, healthy = parse_label(name)
        return Prediction(
            class_name=name,
            plant=plant,
            condition=condition,
            is_healthy=healthy,
            confidence=float(probs[best]),
            top_k=[(self.class_names[i], float(probs[i])) for i in order[:top_k]],
        )

    def predict(self, image: ImageInput, top_k: int = 3) -> Prediction:
        batch = self.preprocess(image)[None, ...]
        probs = self.model.predict(batch, verbose=0)[0]
        return self._to_prediction(probs, top_k)

    def predict_many(self, images: Iterable[ImageInput], top_k: int = 3,
                     batch_size: int = 32) -> list[Prediction]:
        arrays = [self.preprocess(img) for img in images]
        if not arrays:
            return []
        probs = self.model.predict(np.stack(arrays), batch_size=batch_size, verbose=0)
        return [self._to_prediction(p, top_k) for p in probs]
