"""Model architectures.

``cnn`` is the exact architecture from the original notebook. The only
addition is a ``Rescaling(1/255)`` layer at the input, so the saved model
accepts raw 0-255 pixels and callers never have to remember to normalise.

``mobilenetv2`` is an optional transfer-learning model (ImageNet weights),
usually more accurate and smaller, suggested in the README's improvements.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from . import config

ARCHITECTURES = ("cnn", "mobilenetv2")


def _augmentation() -> keras.Sequential:
    """Light augmentation; these layers are only active during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_cnn(num_classes: int, img_size=config.IMG_SIZE, augment: bool = False) -> keras.Model:
    stack = [keras.Input(shape=(*img_size, 3))]
    if augment:
        stack.append(_augmentation())
    stack += [
        layers.Rescaling(1.0 / 255, name="rescale"),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    return keras.Sequential(stack, name="plant_disease_cnn")


def build_mobilenetv2(num_classes: int, img_size=config.IMG_SIZE, augment: bool = False) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # feature extraction; fine-tune later if you like

    inputs = keras.Input(shape=(*img_size, 3))
    x = _augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")


def build_model(arch: str, num_classes: int, img_size=config.IMG_SIZE,
                augment: bool = False, learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    builders = {"cnn": build_cnn, "mobilenetv2": build_mobilenetv2}
    if arch not in builders:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}.")
    model = builders[arch](num_classes, img_size=img_size, augment=augment)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
