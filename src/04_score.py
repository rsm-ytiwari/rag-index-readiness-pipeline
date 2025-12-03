"""
04_score.py - Add quality scores to review data for RAG index readiness.

This script:
- Loads feature-engineered data from Gold layer
- Computes quality scores (informativeness, reliability, relevance)
- Ranks reviews for RAG indexing priority
- Saves scored data back to Gold layer
"""

import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_gold_data(gold_path: str) -> pd.DataFrame:
    """Load feature-engineered data from Gold layer."""
    logger.info(f"Loading data from {gold_path}")

    con = duckdb.connect()
    df = con.execute(f"""
        SELECT * FROM read_parquet('{gold_path}')
    """).df()
    con.close()

    logger.info(f"Loaded {len(df):,} records from Gold")
    return df


def compute_informativeness_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute informativeness score based on text characteristics.
    Higher scores indicate more informative reviews.
    """
    logger.info("Computing informativeness scores...")

    # Components of informativeness
    scaler = MinMaxScaler()

    # Normalized features
    df["norm_word_count"] = scaler.fit_transform(df[["word_count"]])
    df["norm_sentence_count"] = scaler.fit_transform(df[["sentence_count"]])
    df["norm_char_diversity"] = scaler.fit_transform(df[["char_diversity"]])

    # Penalize extremely short or extremely long reviews
    df["length_penalty"] = df["word_count"].apply(
        lambda x: 1.0
        if 50 <= x <= 500
        else 0.7
        if 20 <= x < 50 or 500 < x <= 1000
        else 0.5
    )

    # Informativeness score (weighted combination)
    df["informativeness_score"] = (
        0.35 * df["norm_word_count"]
        + 0.25 * df["norm_sentence_count"]
        + 0.20 * df["norm_char_diversity"]
        + 0.20 * df["length_penalty"]
    )

    # Normalize to 0-1 range
    df["informativeness_score"] = scaler.fit_transform(df[["informativeness_score"]])

    return df


def compute_reliability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reliability score based on user behavior and review metadata.
    Higher scores indicate more reliable reviews.
    """
    logger.info("Computing reliability scores...")

    scaler = MinMaxScaler()

    # User activity (experienced reviewers are more reliable)
    df["norm_user_review_count"] = scaler.fit_transform(df[["user_review_count"]])

    # Social validation (useful votes)
    if "useful" in df.columns:
        df["norm_useful"] = scaler.fit_transform(df[["useful"]])
    else:
        df["norm_useful"] = 0

    # Consistency with business average
    df["consistency_score"] = 1 - (np.abs(df["stars_vs_business_avg"]) / 4)

    # Star rating extremity (extreme ratings might be less reliable)
    df["extremity_penalty"] = df["stars"].apply(lambda x: 0.8 if x in [1, 5] else 1.0)

    # Reliability score (weighted combination)
    df["reliability_score"] = (
        0.30 * df["norm_user_review_count"]
        + 0.25 * df.get("norm_useful", 0)
        + 0.25 * df["consistency_score"]
        + 0.20 * df["extremity_penalty"]
    )

    # Normalize to 0-1 range
    df["reliability_score"] = scaler.fit_transform(df[["reliability_score"]])

    return df


def compute_relevance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relevance score based on recency and business popularity.
    Higher scores indicate more relevant reviews for current queries.
    """
    logger.info("Computing relevance scores...")

    scaler = MinMaxScaler()

    # Recency score (more recent reviews are more relevant)
    if "date" in df.columns and "review_year" in df.columns:
        current_year = datetime.now().year
        df["years_old"] = current_year - df["review_year"]
        # Exponential decay for age
        df["recency_score"] = np.exp(-df["years_old"] / 3)  # Half-life of ~2 years
    else:
        df["recency_score"] = 0.5  # Default if no date

    # Business popularity
    df["norm_business_review_count"] = scaler.fit_transform(
        df[["business_review_count"]]
    )

    # Sentiment alignment with rating
    # Reviews where sentiment matches stars are more relevant
    def sentiment_rating_alignment(row):
        if row["stars"] >= 4 and row["sentiment_ratio"] > 0:
            return 1.0
        elif row["stars"] <= 2 and row["sentiment_ratio"] < 0:
            return 1.0
        elif row["stars"] == 3 and abs(row["sentiment_ratio"]) < 0.3:
            return 1.0
        else:
            return 0.6

    df["sentiment_alignment"] = df.apply(sentiment_rating_alignment, axis=1)

    # Relevance score (weighted combination)
    df["relevance_score"] = (
        0.40 * df["recency_score"]
        + 0.30 * df["norm_business_review_count"]
        + 0.30 * df["sentiment_alignment"]
    )

    # Normalize to 0-1 range
    df["relevance_score"] = scaler.fit_transform(df[["relevance_score"]])

    return df


def compute_overall_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall quality score as weighted combination of all scores.
    This is the primary score for RAG index prioritization.
    """
    logger.info("Computing overall quality scores...")

    # Overall quality score (weighted combination)
    df["quality_score"] = (
        0.35 * df["informativeness_score"]
        + 0.35 * df["reliability_score"]
        + 0.30 * df["relevance_score"]
    )

    # Normalize to 0-100 scale for easier interpretation
    scaler = MinMaxScaler(feature_range=(0, 100))
    df["quality_score"] = scaler.fit_transform(df[["quality_score"]])

    return df


def assign_quality_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign quality tiers for RAG indexing strategy."""
    logger.info("Assigning quality tiers...")

    # Define quality tiers based on percentiles
    df["quality_tier"] = pd.cut(
        df["quality_score"],
        bins=[0, 40, 70, 85, 100],
        labels=["low", "medium", "high", "premium"],
        include_lowest=True,
    )

    # Priority ranking (lower number = higher priority)
    df["rag_priority"] = (
        df["quality_score"].rank(ascending=False, method="dense").astype(int)
    )

    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential outlier reviews for review."""
    logger.info("Flagging potential outliers...")

    # Flag reviews with extremely low quality scores
    df["is_low_quality"] = (df["quality_score"] < 20).astype(int)

    # Flag reviews with mismatched sentiment and rating
    df["is_sentiment_mismatch"] = (
        ((df["stars"] >= 4) & (df["sentiment_ratio"] < -0.3))
        | ((df["stars"] <= 2) & (df["sentiment_ratio"] > 0.3))
    ).astype(int)

    # Flag suspiciously short reviews with high ratings
    df["is_suspicious"] = ((df["word_count"] < 10) & (df["stars"] == 5)).astype(int)

    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add scoring metadata."""
    df["scored_at"] = datetime.now().isoformat()
    df["scoring_version"] = "1.0"

    return df


def save_scored_data(df: pd.DataFrame, output_path: str) -> None:
    """Save scored data back to Gold layer."""
    logger.info(f"Saving {len(df):,} scored records to {output_path}")

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet with compression
    df.to_parquet(output_path, index=False, compression="snappy")

    logger.info(f"Successfully saved scored data")


def print_score_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of scores."""
    logger.info("\n" + "=" * 80)
    logger.info("SCORE SUMMARY")
    logger.info("=" * 80)

    # Quality score distribution
    logger.info(f"\nQuality Score Distribution:")
    logger.info(f"  Mean: {df['quality_score'].mean():.2f}")
    logger.info(f"  Median: {df['quality_score'].median():.2f}")
    logger.info(f"  Std Dev: {df['quality_score'].std():.2f}")
    logger.info(f"  Min: {df['quality_score'].min():.2f}")
    logger.info(f"  Max: {df['quality_score'].max():.2f}")

    # Quality tier distribution
    logger.info(f"\nQuality Tier Distribution:")
    tier_counts = df["quality_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")

    # Component scores
    logger.info(f"\nComponent Scores (Mean):")
    logger.info(f"  Informativeness: {df['informativeness_score'].mean():.3f}")
    logger.info(f"  Reliability: {df['reliability_score'].mean():.3f}")
    logger.info(f"  Relevance: {df['relevance_score'].mean():.3f}")

    # Flags
    logger.info(f"\nQuality Flags:")
    logger.info(
        f"  Low Quality: {df['is_low_quality'].sum():,} ({df['is_low_quality'].sum() / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"  Sentiment Mismatch: {df['is_sentiment_mismatch'].sum():,} ({df['is_sentiment_mismatch'].sum() / len(df) * 100:.1f}%)"
    )
    logger.info(
        f"  Suspicious: {df['is_suspicious'].sum():,} ({df['is_suspicious'].sum() / len(df) * 100:.1f}%)"
    )

    logger.info("=" * 80)


def main():
    """Main scoring pipeline."""
    # Define paths
    base_path = Path(__file__).parent.parent
    gold_input_path = base_path / "data" / "gold" / "reviews_featured.parquet"
    gold_output_path = base_path / "data" / "gold" / "reviews_scored.parquet"

    logger.info("=" * 80)
    logger.info("Starting quality scoring pipeline")
    logger.info("=" * 80)

    try:
        # Load data
        df = load_gold_data(str(gold_input_path))

        # Compute scores
        df = compute_informativeness_score(df)
        df = compute_reliability_score(df)
        df = compute_relevance_score(df)
        df = compute_overall_quality_score(df)
        df = assign_quality_tiers(df)
        df = flag_outliers(df)
        df = add_metadata(df)

        # Print summary
        print_score_summary(df)

        # Save scored data
        save_scored_data(df, str(gold_output_path))

        logger.info("\n" + "=" * 80)
        logger.info("Quality scoring pipeline completed successfully")
        logger.info(f"Output: {len(df):,} scored records")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in scoring pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
