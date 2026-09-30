"""Evaluate a trained model: accuracy, precision, recall, F1, confusion matrix.

Usage
-----
    python evaluate.py                         # evaluate on data/valid
    python evaluate.py --include-train         # also report training-set accuracy (slow)
    python evaluate.py --split-dir some/folder # any folder laid out as <class>/<images>

Outputs
-------
    outputs/metrics.json
    outputs/classification_report.txt
    outputs/confusion_matrix.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from tensorflow import keras

from plant_disease import config
from plant_disease.data import load_class_names, make_dataset
from plant_disease.plots import plot_confusion_matrix


def predict_dataset(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as integer class indices."""
    y_true, y_pred = [], []
    for images, labels in ds:
        probs = model.predict_on_batch(images)
        y_true.append(np.argmax(labels, axis=1))
        y_pred.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--split-dir", type=Path, default=config.VALID_DIR,
                   help="Labelled folder to evaluate on (default: data/valid)")
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--include-train", action="store_true",
                   help="Also compute accuracy on data/train (70k images - takes a while)")
    args = p.parse_args()

    model = keras.models.load_model(str(args.model_dir / config.MODEL_FILENAME), compile=False)
    class_names = load_class_names(args.model_dir)
    img_size = tuple(model.input_shape[1:3])

    ds, ds_classes = make_dataset(args.split_dir, img_size=img_size,
                                  batch_size=args.batch_size, shuffle=False)
    if ds_classes != class_names:
        raise SystemExit(f"Class folders in {args.split_dir} don't match the model's class_names.json")

    print(f"[eval] Predicting on {args.split_dir} ...")
    y_true, y_pred = predict_dataset(model, ds)

    metrics = {
        "split": str(args.split_dir),
        "num_images": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if args.include_train:
        print("[eval] Predicting on training set ...")
        train_ds, _ = make_dataset(config.TRAIN_DIR, img_size=img_size,
                                   batch_size=args.batch_size, shuffle=False)
        t_true, t_pred = predict_dataset(model, train_ds)
        metrics["train_accuracy"] = accuracy_score(t_true, t_pred)

    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                   target_names=class_names, digits=3, zero_division=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}, indent=2))
    (args.output_dir / "classification_report.txt").write_text(report)
    cm_path = plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir / "confusion_matrix.png")

    print("\nClassification report\n" + report)
    print("=" * 60)
    if "train_accuracy" in metrics:
        print(f"Train accuracy     : {metrics['train_accuracy']:.2%}")
    print(f"Accuracy           : {metrics['accuracy']:.2%}")
    print(f"Precision (macro)  : {metrics['precision_macro']:.2%}")
    print(f"Recall (macro)     : {metrics['recall_macro']:.2%}")
    print(f"F1 (macro)         : {metrics['f1_macro']:.2%}")
    print(f"Saved: {args.output_dir / 'metrics.json'}, classification_report.txt, {cm_path.name}")


if __name__ == "__main__":
    main()
