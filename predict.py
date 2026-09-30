"""Predict plant disease / health for one or more leaf images.

Usage
-----
    python predict.py leaf.jpg                      # one image
    python predict.py img1.jpg img2.png some_folder # several images and/or folders
    python predict.py                               # everything in data/test
    python predict.py data/test --grid              # also save outputs/predictions_grid.png
    python predict.py leaf.jpg --top-k 5            # show 5 most likely classes
"""

import argparse
from pathlib import Path

from plant_disease import config
from plant_disease.inference import PlantDiseasePredictor, parse_label


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for path in paths:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        elif path.is_file():
            images.append(path)
        else:
            print(f"[predict] Skipping (not found): {path}")
    return images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, default=[config.TEST_DIR],
                   help="Image files and/or folders (default: data/test)")
    p.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--grid", nargs="?", type=Path, const=config.OUTPUT_DIR / "predictions_grid.png",
                   default=None, help="Save a 3x3 grid of predictions (optional output path)")
    args = p.parse_args()

    images = collect_images(args.paths)
    if not images:
        raise SystemExit("No images found. Pass an image path or a folder of images.")

    predictor = PlantDiseasePredictor.from_dir(args.model_dir)
    predictions = predictor.predict_many(images, top_k=args.top_k)

    for path, pred in zip(images, predictions):
        print(f"\n{path.name}")
        print(f"  {pred.summary}")
        if args.top_k > 1:
            for name, prob in pred.top_k:
                plant, condition, _ = parse_label(name)
                print(f"     {prob:6.1%}  {plant} - {condition}")

    healthy = sum(p.is_healthy for p in predictions)
    print(f"\n{len(predictions)} image(s): {healthy} healthy, {len(predictions) - healthy} unhealthy")

    if args.grid:
        from plant_disease.plots import plot_prediction_grid
        out = plot_prediction_grid(images, predictions, args.grid)
        print(f"Grid saved to {out}")


if __name__ == "__main__":
    main()
