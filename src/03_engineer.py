"""
03_engineer.py - Feature engineering for review data (Silver → Gold layer).

This script:
- Loads cleaned data from Silver layer
- Computes text features (length, readability, sentiment indicators)
- Calculates review metadata features
- Adds temporal features
- Saves feature-engineered data to Gold layer
"""

import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from utils import compute_text_stats, chunk_text

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_silver_data(silver_path: str) -> pd.DataFrame:
    """Load cleaned data from Silver layer."""
    logger.info(f"Loading data from {silver_path}")

    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_parquet('{silver_path}')
    """).df()
    con.close()

    logger.info(f"Loaded {len(df):,} records from Silver")
    return df


def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute text-based features."""
    logger.info("Computing text features...")

    # Get text statistics
    text_stats = df["text_cleaned"].apply(compute_text_stats)

    df["word_count"] = text_stats.apply(lambda x: x["word_count"])
    df["sentence_count"] = text_stats.apply(lambda x: x["sentence_count"])
    df["avg_word_length"] = text_stats.apply(lambda x: x["avg_word_length"])

    # Words per sentence (readability indicator)
    df["words_per_sentence"] = df["word_count"] / df["sentence_count"].replace(0, 1)

    # Character diversity
    df["char_diversity"] = df["text_cleaned"].apply(
        lambda x: len(set(x.lower())) / len(x) if len(x) > 0 else 0
    )

    return df


def compute_sentiment_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple sentiment indicators (lexicon-based)."""
    logger.info("Computing sentiment indicators...")

    # Positive words (simple lexicon)
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "best",
        "perfect",
        "awesome",
        "delicious",
        "friendly",
        "clean",
        "recommend",
        "favorite",
        "enjoyed",
        "nice",
        "beautiful",
    }

    # Negative words (simple lexicon)
    negative_words = {
        "bad",
        "terrible",
        "horrible",
        "awful",
        "worst",
        "disgusting",
        "hate",
        "poor",
        "disappointing",
        "rude",
        "dirty",
        "slow",
        "avoid",
        "never",
        "waste",
        "overpriced",
        "mediocre",
        "tasteless",
    }

    def count_sentiment_words(text, word_set):
        text_lower = text.lower()
        words = text_lower.split()
        return sum(1 for word in words if word in word_set)

    df["positive_word_count"] = df["text_cleaned"].apply(
        lambda x: count_sentiment_words(x, positive_words)
    )
    df["negative_word_count"] = df["text_cleaned"].apply(
        lambda x: count_sentiment_words(x, negative_words)
    )

    # Sentiment ratio
    total_sentiment = df["positive_word_count"] + df["negative_word_count"]
    df["sentiment_ratio"] = np.where(
        total_sentiment > 0,
        (df["positive_word_count"] - df["negative_word_count"]) / total_sentiment,
        0,
    )

    return df


def compute_review_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metadata-based features."""
    logger.info("Computing metadata features...")

    # Convert date if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        # Temporal features
        df["review_year"] = df["date"].dt.year
        df["review_month"] = df["date"].dt.month
        df["review_day_of_week"] = df["date"].dt.dayofweek
        df["review_hour"] = df["date"].dt.hour if "hour" in df["date"].dt else None

    # Usefulness indicators (if columns exist)
    if "useful" in df.columns:
        df["has_useful_votes"] = (df["useful"] > 0).astype(int)

    if "funny" in df.columns:
        df["has_funny_votes"] = (df["funny"] > 0).astype(int)

    if "cool" in df.columns:
        df["has_cool_votes"] = (df["cool"] > 0).astype(int)

    return df


def compute_business_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute business-level aggregate features."""
    logger.info("Computing business aggregate features...")

    # Reviews per business
    business_counts = df.groupby("business_id").size().rename("business_review_count")
    df = df.merge(business_counts, left_on="business_id", right_index=True, how="left")

    # Average stars per business
    business_avg_stars = (
        df.groupby("business_id")["stars"].mean().rename("business_avg_stars")
    )
    df = df.merge(
        business_avg_stars, left_on="business_id", right_index=True, how="left"
    )

    # Star deviation from business average
    df["stars_vs_business_avg"] = df["stars"] - df["business_avg_stars"]

    return df


def compute_user_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute user-level aggregate features."""
    logger.info("Computing user aggregate features...")

    # Reviews per user
    user_counts = df.groupby("user_id").size().rename("user_review_count")
    df = df.merge(user_counts, left_on="user_id", right_index=True, how="left")

    # Average stars per user
    user_avg_stars = df.groupby("user_id")["stars"].mean().rename("user_avg_stars")
    df = df.merge(user_avg_stars, left_on="user_id", right_index=True, how="left")

    # User's average review length
    user_avg_length = (
        df.groupby("user_id")["text_length"].mean().rename("user_avg_text_length")
    )
    df = df.merge(user_avg_length, left_on="user_id", right_index=True, how="left")

    return df


def create_text_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Create text chunks for RAG retrieval."""
    logger.info("Creating text chunks for RAG...")

    # Create chunks for longer reviews
    df["text_chunks"] = df["text_cleaned"].apply(
        lambda x: chunk_text(x, chunk_size=512, overlap=50)
    )
    df["chunk_count"] = df["text_chunks"].apply(len)

    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add processing metadata."""
    df["engineered_at"] = datetime.now().isoformat()
    df["processing_stage"] = "gold"

    return df


def save_to_gold(df: pd.DataFrame, gold_path: str) -> None:
    """Save feature-engineered data to Gold layer."""
    logger.info(f"Saving {len(df):,} records to {gold_path}")

    # Ensure directory exists
    Path(gold_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert list columns to string for Parquet compatibility
    if "text_chunks" in df.columns:
        df["text_chunks"] = df["text_chunks"].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )

    # Save as Parquet with compression
    df.to_parquet(gold_path, index=False, compression="snappy")

    logger.info(f"Successfully saved to Gold layer")


def main():
    """Main feature engineering pipeline."""
    # Define paths
    base_path = Path(__file__).parent.parent
    silver_path = base_path / "data" / "Silver" / "reviews_cleaned.parquet"
    gold_path = base_path / "data" / "gold" / "reviews_featured.parquet"

    logger.info("=" * 80)
    logger.info("Starting feature engineering pipeline (Silver → Gold)")
    logger.info("=" * 80)

    try:
        # Load data
        df = load_silver_data(str(silver_path))

        # Feature engineering
        df = compute_text_features(df)
        df = compute_sentiment_indicators(df)
        df = compute_review_metadata_features(df)
        df = compute_business_aggregates(df)
        df = compute_user_aggregates(df)
        df = create_text_chunks(df)
        df = add_metadata(df)

        # Save to Gold
        save_to_gold(df, str(gold_path))

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("Feature engineering pipeline completed successfully")
        logger.info(f"Output: {len(df):,} records with {len(df.columns)} features")
        logger.info(f"Average chunks per review: {df['chunk_count'].mean():.2f}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
