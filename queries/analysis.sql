-- analysis.sql - DuckDB queries for review data analytics
-- This file contains SQL queries for analyzing the RAG-ready review data

-- ============================================================================
-- DATA QUALITY ANALYSIS
-- ============================================================================

-- Overview of data quality across pipeline stages
CREATE OR REPLACE VIEW data_quality_overview AS
SELECT
    processing_stage,
    COUNT(*) as record_count,
    AVG(text_length) as avg_text_length,
    SUM(CASE WHEN has_pii THEN 1 ELSE 0 END) as pii_count,
    COUNT(DISTINCT business_id) as unique_businesses,
    COUNT(DISTINCT user_id) as unique_users
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY processing_stage;

-- Quality tier distribution
CREATE OR REPLACE VIEW quality_tier_summary AS
SELECT
    quality_tier,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(stars), 2) as avg_stars,
    ROUND(AVG(word_count), 0) as avg_word_count
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY quality_tier
ORDER BY
    CASE quality_tier
        WHEN 'premium' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END;

-- ============================================================================
-- BUSINESS ANALYSIS
-- ============================================================================

-- Top businesses by review count and quality
CREATE OR REPLACE VIEW top_businesses AS
SELECT
    business_id,
    COUNT(*) as review_count,
    ROUND(AVG(stars), 2) as avg_stars,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(word_count), 0) as avg_review_length,
    COUNT(CASE WHEN quality_tier IN ('high', 'premium') THEN 1 END) as high_quality_reviews
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY business_id
HAVING COUNT(*) >= 10
ORDER BY avg_quality_score DESC, review_count DESC
LIMIT 100;

-- Business review distribution by star rating
CREATE OR REPLACE VIEW business_star_distribution AS
SELECT
    business_id,
    SUM(CASE WHEN stars = 5 THEN 1 ELSE 0 END) as five_star,
    SUM(CASE WHEN stars = 4 THEN 1 ELSE 0 END) as four_star,
    SUM(CASE WHEN stars = 3 THEN 1 ELSE 0 END) as three_star,
    SUM(CASE WHEN stars = 2 THEN 1 ELSE 0 END) as two_star,
    SUM(CASE WHEN stars = 1 THEN 1 ELSE 0 END) as one_star,
    COUNT(*) as total_reviews
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY business_id
HAVING COUNT(*) >= 5;

-- ============================================================================
-- USER ANALYSIS
-- ============================================================================

-- Top reviewers by activity and quality
CREATE OR REPLACE VIEW top_reviewers AS
SELECT
    user_id,
    COUNT(*) as review_count,
    ROUND(AVG(stars), 2) as avg_stars,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(word_count), 0) as avg_review_length,
    ROUND(AVG(reliability_score), 3) as avg_reliability
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY user_id
HAVING COUNT(*) >= 5
ORDER BY avg_quality_score DESC, review_count DESC
LIMIT 100;

-- User engagement patterns
CREATE OR REPLACE VIEW user_engagement_patterns AS
SELECT
    CASE
        WHEN user_review_count = 1 THEN '1 review'
        WHEN user_review_count BETWEEN 2 AND 5 THEN '2-5 reviews'
        WHEN user_review_count BETWEEN 6 AND 10 THEN '6-10 reviews'
        WHEN user_review_count BETWEEN 11 AND 20 THEN '11-20 reviews'
        WHEN user_review_count BETWEEN 21 AND 50 THEN '21-50 reviews'
        ELSE '50+ reviews'
    END as user_segment,
    COUNT(DISTINCT user_id) as user_count,
    COUNT(*) as total_reviews,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(reliability_score), 3) as avg_reliability_score
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY user_segment
ORDER BY
    CASE user_segment
        WHEN '1 review' THEN 1
        WHEN '2-5 reviews' THEN 2
        WHEN '6-10 reviews' THEN 3
        WHEN '11-20 reviews' THEN 4
        WHEN '21-50 reviews' THEN 5
        ELSE 6
    END;

-- ============================================================================
-- TEXT ANALYSIS
-- ============================================================================

-- Review length distribution and quality correlation
CREATE OR REPLACE VIEW text_length_analysis AS
SELECT
    CASE
        WHEN word_count < 20 THEN 'Very Short (<20)'
        WHEN word_count BETWEEN 20 AND 50 THEN 'Short (20-50)'
        WHEN word_count BETWEEN 51 AND 100 THEN 'Medium (51-100)'
        WHEN word_count BETWEEN 101 AND 200 THEN 'Long (101-200)'
        WHEN word_count BETWEEN 201 AND 500 THEN 'Very Long (201-500)'
        ELSE 'Extremely Long (>500)'
    END as length_category,
    COUNT(*) as review_count,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(informativeness_score), 3) as avg_informativeness,
    ROUND(AVG(stars), 2) as avg_stars
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY length_category
ORDER BY
    CASE length_category
        WHEN 'Very Short (<20)' THEN 1
        WHEN 'Short (20-50)' THEN 2
        WHEN 'Medium (51-100)' THEN 3
        WHEN 'Long (101-200)' THEN 4
        WHEN 'Very Long (201-500)' THEN 5
        ELSE 6
    END;

-- Sentiment analysis summary
CREATE OR REPLACE VIEW sentiment_analysis AS
SELECT
    stars,
    COUNT(*) as review_count,
    ROUND(AVG(sentiment_ratio), 3) as avg_sentiment_ratio,
    ROUND(AVG(positive_word_count), 1) as avg_positive_words,
    ROUND(AVG(negative_word_count), 1) as avg_negative_words,
    SUM(CASE WHEN is_sentiment_mismatch = 1 THEN 1 ELSE 0 END) as mismatch_count
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY stars
ORDER BY stars DESC;

-- ============================================================================
-- TEMPORAL ANALYSIS
-- ============================================================================

-- Review trends over time (if date available)
CREATE OR REPLACE VIEW temporal_trends AS
SELECT
    review_year,
    review_month,
    COUNT(*) as review_count,
    ROUND(AVG(stars), 2) as avg_stars,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(word_count), 0) as avg_word_count
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE review_year IS NOT NULL
GROUP BY review_year, review_month
ORDER BY review_year DESC, review_month DESC;

-- Day of week patterns
CREATE OR REPLACE VIEW day_of_week_patterns AS
SELECT
    review_day_of_week,
    CASE review_day_of_week
        WHEN 0 THEN 'Monday'
        WHEN 1 THEN 'Tuesday'
        WHEN 2 THEN 'Wednesday'
        WHEN 3 THEN 'Thursday'
        WHEN 4 THEN 'Friday'
        WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END as day_name,
    COUNT(*) as review_count,
    ROUND(AVG(stars), 2) as avg_stars,
    ROUND(AVG(word_count), 0) as avg_word_count
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE review_day_of_week IS NOT NULL
GROUP BY review_day_of_week
ORDER BY review_day_of_week;

-- ============================================================================
-- RAG INDEX READINESS
-- ============================================================================

-- Premium reviews for RAG indexing (top priority)
CREATE OR REPLACE VIEW rag_premium_reviews AS
SELECT
    review_id,
    business_id,
    user_id,
    stars,
    text_cleaned,
    quality_score,
    quality_tier,
    rag_priority,
    chunk_count,
    word_count,
    informativeness_score,
    reliability_score,
    relevance_score
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE quality_tier = 'premium'
ORDER BY rag_priority;

-- RAG indexing strategy summary
CREATE OR REPLACE VIEW rag_indexing_strategy AS
SELECT
    quality_tier,
    COUNT(*) as review_count,
    SUM(chunk_count) as total_chunks,
    ROUND(AVG(chunk_count), 1) as avg_chunks_per_review,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage_of_corpus
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY quality_tier
ORDER BY
    CASE quality_tier
        WHEN 'premium' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END;

-- Reviews flagged for review (quality issues)
CREATE OR REPLACE VIEW flagged_reviews AS
SELECT
    review_id,
    business_id,
    stars,
    word_count,
    quality_score,
    is_low_quality,
    is_sentiment_mismatch,
    is_suspicious,
    sentiment_ratio,
    text_cleaned
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE is_low_quality = 1
   OR is_sentiment_mismatch = 1
   OR is_suspicious = 1
ORDER BY quality_score;

-- ============================================================================
-- SCORE CORRELATIONS
-- ============================================================================

-- Correlation between scores and star ratings
CREATE OR REPLACE VIEW score_correlations AS
SELECT
    stars,
    COUNT(*) as review_count,
    ROUND(AVG(quality_score), 2) as avg_quality_score,
    ROUND(AVG(informativeness_score), 3) as avg_informativeness,
    ROUND(AVG(reliability_score), 3) as avg_reliability,
    ROUND(AVG(relevance_score), 3) as avg_relevance
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY stars
ORDER BY stars DESC;

-- ============================================================================
-- EXAMPLE QUERIES FOR DATA EXPLORATION
-- ============================================================================

-- Find high-quality detailed reviews for a specific business
-- USAGE: Replace 'BUSINESS_ID' with actual business_id
/*
SELECT
    review_id,
    stars,
    text_cleaned,
    quality_score,
    word_count,
    sentiment_ratio
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE business_id = 'BUSINESS_ID'
  AND quality_tier IN ('high', 'premium')
ORDER BY quality_score DESC
LIMIT 10;
*/

-- Compare review quality across star ratings
-- USAGE: Run as-is
/*
SELECT
    stars,
    quality_tier,
    COUNT(*) as review_count,
    ROUND(AVG(word_count), 0) as avg_word_count,
    ROUND(AVG(quality_score), 2) as avg_quality_score
FROM read_parquet('data/gold/reviews_scored.parquet')
GROUP BY stars, quality_tier
ORDER BY stars DESC,
    CASE quality_tier
        WHEN 'premium' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END;
*/

-- Find reviews with highest informativeness for RAG
-- USAGE: Adjust LIMIT as needed
/*
SELECT
    review_id,
    business_id,
    stars,
    quality_score,
    informativeness_score,
    word_count,
    chunk_count,
    text_cleaned
FROM read_parquet('data/gold/reviews_scored.parquet')
WHERE quality_tier IN ('high', 'premium')
ORDER BY informativeness_score DESC
LIMIT 20;
*/
