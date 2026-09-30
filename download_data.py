"""Download the Kaggle dataset and arrange it as data/train, data/valid, data/test.

Usage
-----
    python download_data.py                      # download via Kaggle API
    python download_data.py --zip path/to.zip    # use a zip you downloaded manually
    python download_data.py --cleanup            # also delete the zip + raw folder afterwards
"""

import argparse
import shutil
from pathlib import Path

from plant_disease import config
from plant_disease.data import (count_images, download_from_kaggle, extract_zip,
                                organise_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=None,
                        help="Path to an already-downloaded dataset zip (skips the Kaggle API).")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the zip and the leftover raw folder when done (saves ~3 GB).")
    args = parser.parse_args()

    zip_path = args.zip if args.zip else download_from_kaggle()
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    extract_zip(zip_path, config.RAW_DIR)
    organise_dataset(config.RAW_DIR)

    print("\n[data] Done:")
    for split in (config.TRAIN_DIR, config.VALID_DIR, config.TEST_DIR):
        if split.exists():
            print(f"  {split.relative_to(config.PROJECT_ROOT)}: {count_images(split):,} images")

    if args.cleanup:
        shutil.rmtree(config.RAW_DIR, ignore_errors=True)
        if args.zip is None:  # only delete a zip we downloaded ourselves
            zip_path.unlink(missing_ok=True)
        print("[data] Cleaned up raw files.")


if __name__ == "__main__":
    main()
