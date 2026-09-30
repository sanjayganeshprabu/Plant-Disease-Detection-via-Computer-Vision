"""Train the plant disease classifier.

Usage
-----
    python train.py                                   # original CNN, defaults from config.py
    python train.py --epochs 30 --augment             # CNN + data augmentation
    python train.py --arch mobilenetv2 --augment      # transfer learning (usually better)

Outputs
-------
    models/plant_disease_model.keras   trained model (best epoch by val_loss)
    models/class_names.json            label order, needed for inference
    models/training_history.json       per-epoch metrics
    outputs/training_curves.png        loss / accuracy plots
    outputs/training_log.csv           same metrics as CSV
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from plant_disease import config
from plant_disease.data import make_dataset, save_class_names
from plant_disease.model import ARCHITECTURES, build_model
from plant_disease.plots import plot_training_history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                   help="Folder containing train/ and valid/ (default: data/)")
    p.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--arch", choices=ARCHITECTURES, default="cnn")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE[0],
                   help="Square image size in pixels (default: 128)")
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE,
                   help="Early-stopping patience in epochs")
    p.add_argument("--augment", action="store_true", help="Enable random flip/rotate/zoom/contrast")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    img_size = (args.img_size, args.img_size)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[train] TensorFlow {tf.__version__} | GPUs: {len(gpus) or 'none (training on CPU will be slow)'}")

    # Training data is shuffled every epoch (the original notebook used
    # shuffle=False, which feeds the network one class at a time).
    train_ds, class_names = make_dataset(args.data_dir / "train", img_size=img_size,
                                         batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_ds, val_classes = make_dataset(args.data_dir / "valid", img_size=img_size,
                                       batch_size=args.batch_size, shuffle=False)
    if class_names != val_classes:
        raise SystemExit("train/ and valid/ contain different class folders.")

    model = build_model(args.arch, num_classes=len(class_names), img_size=img_size,
                        augment=args.augment, learning_rate=args.lr)
    model.summary()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = args.model_dir / "checkpoints" / "best.weights.h5"
    best_weights.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1),
        keras.callbacks.ModelCheckpoint(str(best_weights), monitor="val_loss",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(1, args.patience - 1), verbose=1),
        keras.callbacks.CSVLogger(str(args.output_dir / "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)

    # Always finish with the best epoch's weights, whether or not early stopping fired.
    model.load_weights(str(best_weights))
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)

    model_path = args.model_dir / config.MODEL_FILENAME
    model.save(str(model_path))
    save_class_names(class_names, args.model_dir)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist["config"] = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      "best_val_loss": float(val_loss), "best_val_accuracy": float(val_acc)}
    (args.model_dir / config.HISTORY_FILENAME).write_text(json.dumps(hist, indent=2))
    curves = plot_training_history(history.history, args.output_dir / "training_curves.png")

    print("\n" + "=" * 60)
    print(f"Best validation accuracy : {val_acc:.2%}   (loss {val_loss:.4f})")
    print(f"Model saved to           : {model_path}")
    print(f"Training curves          : {curves}")
    print("Next: python evaluate.py   |   python predict.py <image>   |   streamlit run app.py")


if __name__ == "__main__":
    main()
