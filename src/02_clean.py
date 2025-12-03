"""
02_clean.py - Clean and validate review data from Bronze to Silver layer.

This script:
- Loads raw review data from Bronze (Parquet)
- Removes duplicates and invalid records
- Cleans text content
- Detects and masks PII
- Saves cleaned data to Silver layer
"""

import pandas as pd
import duckdb
from pathlib import Path
import logging
from datetime import datetime
from utils import clean_text, detect_pii, remove_pii, validate_review_data

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_bronze_data(bronze_path: str) -> pd.DataFrame:
    """Load review data from Bronze layer."""
    logger.info(f"Loading data from {bronze_path}")

    # Use DuckDB for efficient Parquet reading
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_parquet('{bronze_path}')
    """).df()
    con.close()

    logger.info(f"Loaded {len(df):,} records from Bronze")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate reviews based on review_id."""
    initial_count = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    removed_count = initial_count - len(df)

    if removed_count > 0:
        logger.info(f"Removed {removed_count:,} duplicate records")

    return df


def filter_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid records."""
    initial_count = len(df)

    # Remove records with missing critical fields
    df = df[df["text"].notna()]
    df = df[df["stars"].notna()]
    df = df[df["business_id"].notna()]
    df = df[df["user_id"].notna()]

    # Remove records with empty text
    df = df[df["text"].str.strip().str.len() > 0]

    # Remove records with invalid star ratings
    df = df[(df["stars"] >= 1) & (df["stars"] <= 5)]

    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count:,} invalid records")

    return df


def clean_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text content in reviews."""
    logger.info("Cleaning review text...")

    # Clean text
    df["text_cleaned"] = df["text"].apply(clean_text)

    # Calculate text length after cleaning
    df["text_length"] = df["text_cleaned"].str.len()

    # Remove reviews that became too short after cleaning (< 10 characters)
    initial_count = len(df)
    df = df[df["text_length"] >= 10]
    removed_count = initial_count - len(df)

    if removed_count > 0:
        logger.info(f"Removed {removed_count:,} reviews with insufficient text")

    return df


def handle_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and mask PII in review text."""
    logger.info("Detecting and masking PII...")

    # Detect PII
    pii_flags = df["text_cleaned"].apply(detect_pii)
    df["has_pii"] = pii_flags.apply(lambda x: any(x.values()))

    # Log PII statistics
    pii_count = df["has_pii"].sum()
    if pii_count > 0:
        logger.warning(
            f"Found potential PII in {pii_count:,} reviews ({pii_count / len(df) * 100:.2f}%)"
        )

    # Mask PII
    df["text_cleaned"] = df["text_cleaned"].apply(remove_pii)

    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add processing metadata."""
    df["cleaned_at"] = datetime.now().isoformat()
    df["processing_stage"] = "silver"

    return df


def save_to_silver(df: pd.DataFrame, silver_path: str) -> None:
    """Save cleaned data to Silver layer."""
    logger.info(f"Saving {len(df):,} records to {silver_path}")

    # Ensure directory exists
    Path(silver_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet with compression
    df.to_parquet(silver_path, index=False, compression="snappy")

    logger.info(f"Successfully saved to Silver layer")


def main():
    """Main cleaning pipeline."""
    # Define paths
    base_path = Path(__file__).parent.parent
    bronze_path = base_path / "data" / "Bronze" / "reviews_100k.parquet"
    silver_path = base_path / "data" / "Silver" / "reviews_cleaned.parquet"

    logger.info("=" * 80)
    logger.info("Starting data cleaning pipeline (Bronze → Silver)")
    logger.info("=" * 80)

    try:
        # Load data
        df = load_bronze_data(str(bronze_path))

        # Validate initial data
        logger.info("Initial data validation:")
        validation = validate_review_data(df)
        for key, value in validation.items():
            logger.info(f"  {key}: {value}")

        # Clean data
        df = remove_duplicates(df)
        df = filter_invalid_records(df)
        df = clean_review_text(df)
        df = handle_pii(df)
        df = add_metadata(df)

        # Final validation
        logger.info("\nFinal data validation:")
        validation = validate_review_data(df)
        for key, value in validation.items():
            logger.info(f"  {key}: {value}")

        # Save to Silver
        save_to_silver(df, str(silver_path))

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("Cleaning pipeline completed successfully")
        logger.info(f"Output: {len(df):,} cleaned records")
        logger.info(
            f"Data quality: {len(df) / validation.get('total_records', 1) * 100:.2f}% records retained"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in cleaning pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
