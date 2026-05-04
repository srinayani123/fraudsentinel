"""
Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Prerequisites:
1. Kaggle account with API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env)
2. Accept competition rules at https://www.kaggle.com/c/ieee-fraud-detection/rules

Files downloaded into data/raw/:
- train_transaction.csv  (~590k rows, 394 cols)
- train_identity.csv     (~144k rows, 41 cols)
- test_transaction.csv
- test_identity.csv
"""

import os
import zipfile
from pathlib import Path

from src.utils.config import KAGGLE_COMPETITION, RAW_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_kaggle_credentials():
    """Check Kaggle credentials are configured before attempting download."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not kaggle_json.exists() and not has_env:
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Either:\n"
            "  1. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)\n"
            "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
            "Get a token at https://www.kaggle.com/settings → 'Create New API Token'"
        )


def download_competition_data():
    ensure_kaggle_credentials()

    # Import here so missing kaggle package doesn't break the rest of the project
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading {KAGGLE_COMPETITION} into {RAW_DIR}...")
    logger.info("This is ~1.3 GB. May take several minutes.")

    api.competition_download_files(
        competition=KAGGLE_COMPETITION,
        path=str(RAW_DIR),
        quiet=False,
    )

    # Unzip
    zip_path = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        logger.info("Extraction complete.")

    # List downloaded files
    logger.info("Downloaded files:")
    for f in sorted(RAW_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.1f} MB")


def main():
    expected_files = ["train_transaction.csv", "train_identity.csv"]
    if all((RAW_DIR / f).exists() for f in expected_files):
        logger.info("✅ IEEE-CIS data already present. Skipping download.")
        logger.info("   Delete data/raw/ to re-download.")
        return

    try:
        download_competition_data()
        logger.info("✅ Download complete.")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error(
            "If you see a 403 error, accept the competition rules at "
            "https://www.kaggle.com/c/ieee-fraud-detection/rules first."
        )
        raise


if __name__ == "__main__":
    main()
