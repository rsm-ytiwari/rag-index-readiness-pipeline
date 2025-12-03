-- ============================================================================
-- RAG INDEX-READINESS ANALYSIS QUERIES
-- ============================================================================
-- 
-- These queries analyze the Yelp reviews dataset for RAG system readiness.
-- Run with DuckDB on: data/gold/reviews_featured.parquet
--
-- Usage:
--   python -c "import duckdb; con = duckdb.connect(); 
--   con.execute('SELECT * FROM read_parquet(\"data/gold/reviews_featured.parquet\") LIMIT 5').fetchdf()"
-- ============================================================================

-- ============================================================================
-- QUERY 1: OVERALL STATISTICS
-- ============================================================================
-- Purpose: High-level summary of the entire dataset
-- Output: Total reviews, avg score, index-ready %, duplicate %, PII %

SELECT 
    COUNT(*) as total_reviews,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(AVG(token_count), 0) as avg_tokens,
    ROUND(SUM(CASE WHEN index_ready THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_index_ready,
    ROUND(SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_duplicates,
    ROUND(SUM(CASE WHEN has_pii THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_has_pii,
    ROUND(SUM(CASE WHEN chunk_quality_flag = 'optimal' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_optimal_chunks
FROM read_parquet('data/gold/reviews_featured.parquet');


-- ============================================================================
-- QUERY 2: QUALITY DISTRIBUTION
-- ============================================================================
-- Purpose: Breakdown of reviews by quality tier (Index/Review/Reject)
-- Output: Count and percentage for each recommendation tier

SELECT 
    recommendation,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('data/gold/reviews_featured.parquet')), 1) as percentage,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(AVG(token_count), 0) as avg_tokens
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY recommendation
ORDER BY 
    CASE 
        WHEN recommendation = 'index' THEN 1
        WHEN recommendation = 'review' THEN 2
        ELSE 3
    END;


-- ============================================================================
-- QUERY 3: TOP 10 CITIES BY QUALITY SCORE
-- ============================================================================
-- Purpose: Identify cities with highest quality reviews (for targeting)
-- Output: Top 10 cities with avg score, review count, index-ready %
-- Note: Requires business location data - this is a placeholder

-- This query would require joining with business data (not in our dataset)
-- Placeholder query showing what it would look like:

SELECT 
    'Phoenix' as city,
    COUNT(*) as review_count,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(SUM(CASE WHEN index_ready THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_index_ready
FROM read_parquet('data/gold/reviews_featured.parquet')
WHERE review_id LIKE 'A%'  -- Placeholder filter
GROUP BY city
LIMIT 10;

-- Alternative: Top 10 by review_id prefix (as proxy for location)
SELECT 
    SUBSTRING(review_id, 1, 1) as id_prefix,
    COUNT(*) as review_count,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(SUM(CASE WHEN index_ready THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_index_ready,
    ROUND(AVG(token_count), 0) as avg_tokens
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY id_prefix
ORDER BY review_count DESC
LIMIT 10;


-- ============================================================================
-- QUERY 4: PII BREAKDOWN
-- ============================================================================
-- Purpose: Analyze PII detection patterns
-- Output: PII types, frequency, affected review counts

SELECT 
    has_pii,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('data/gold/reviews_featured.parquet')), 1) as percentage,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(AVG(token_count), 0) as avg_tokens
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY has_pii
ORDER BY has_pii DESC;

-- Detailed PII types (if we parse the JSON field)
-- Note: This shows the approach - actual parsing would need JSON functions

SELECT 
    'With PII' as category,
    COUNT(*) as count,
    ROUND(AVG(index_readiness_score), 1) as avg_score
FROM read_parquet('data/gold/reviews_featured.parquet')
WHERE has_pii = true
UNION ALL
SELECT 
    'Without PII' as category,
    COUNT(*) as count,
    ROUND(AVG(index_readiness_score), 1) as avg_score
FROM read_parquet('data/gold/reviews_featured.parquet')
WHERE has_pii = false;


-- ============================================================================
-- QUERY 5: DUPLICATE TREND OVER TIME
-- ============================================================================
-- Purpose: Track duplicate rate by month (for monitoring data quality)
-- Output: Monthly duplicate rates
-- Note: Our dataset has 0% duplicates, so this is a template

SELECT 
    DATE_TRUNC('month', CAST(date AS TIMESTAMP)) as month,
    COUNT(*) as total_reviews,
    SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) as duplicate_count,
    ROUND(SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as duplicate_rate,
    ROUND(AVG(index_readiness_score), 1) as avg_score
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY month
ORDER BY month DESC
LIMIT 24;  -- Last 24 months


-- ============================================================================
-- QUERY 6: CHUNK QUALITY ANALYSIS
-- ============================================================================
-- Purpose: Understand chunk size distribution and quality
-- Output: Chunk quality breakdown with statistics

SELECT 
    chunk_quality_flag,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('data/gold/reviews_featured.parquet')), 1) as percentage,
    ROUND(AVG(chunk_count), 1) as avg_chunks,
    ROUND(AVG(avg_chunk_tokens), 0) as avg_chunk_size,
    ROUND(AVG(index_readiness_score), 1) as avg_score
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY chunk_quality_flag
ORDER BY review_count DESC;


-- ============================================================================
-- QUERY 7: STAR RATING VS QUALITY SCORE
-- ============================================================================
-- Purpose: Correlation between star rating and index-readiness
-- Output: Does review sentiment affect data quality?

SELECT 
    stars,
    COUNT(*) as review_count,
    ROUND(AVG(index_readiness_score), 1) as avg_quality_score,
    ROUND(AVG(token_count), 0) as avg_tokens,
    ROUND(SUM(CASE WHEN index_ready THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_index_ready
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY stars
ORDER BY stars;


-- ============================================================================
-- QUERY 8: PROBLEMATIC REVIEWS (Needs Manual Review)
-- ============================================================================
-- Purpose: Identify reviews requiring human attention
-- Output: Reviews with score < 70, sorted by severity

SELECT 
    review_id,
    stars,
    token_count,
    chunk_quality_flag,
    has_pii,
    is_duplicate,
    index_readiness_score,
    SUBSTRING(text, 1, 100) as text_preview
FROM read_parquet('data/gold/reviews_featured.parquet')
WHERE index_ready = false
ORDER BY index_readiness_score ASC
LIMIT 20;


-- ============================================================================
-- QUERY 9: TOKEN LENGTH DISTRIBUTION
-- ============================================================================
-- Purpose: Understand review length distribution
-- Output: Token count buckets with quality metrics

SELECT 
    CASE 
        WHEN token_count < 50 THEN '0-50 (Very Short)'
        WHEN token_count < 100 THEN '50-100 (Short)'
        WHEN token_count < 200 THEN '100-200 (Medium)'
        WHEN token_count < 500 THEN '200-500 (Long)'
        ELSE '500+ (Very Long)'
    END as token_range,
    COUNT(*) as review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM read_parquet('data/gold/reviews_featured.parquet')), 1) as percentage,
    ROUND(AVG(index_readiness_score), 1) as avg_score,
    ROUND(SUM(CASE WHEN index_ready THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_index_ready
FROM read_parquet('data/gold/reviews_featured.parquet')
GROUP BY token_range
ORDER BY MIN(token_count);


-- ============================================================================
-- QUERY 10: DATA QUALITY SUMMARY FOR REPORTING
-- ============================================================================
-- Purpose: Executive summary for stakeholders
-- Output: Key metrics in single row for dashboard

SELECT 
    COUNT(*) as total_reviews,
    ROUND(AVG(index_readiness_score), 1) as avg_quality_score,
    SUM(CASE WHEN recommendation = 'index' THEN 1 ELSE 0 END) as ready_to_index,
    SUM(CASE WHEN recommendation = 'review' THEN 1 ELSE 0 END) as needs_review,
    SUM(CASE WHEN recommendation = 'reject' THEN 1 ELSE 0 END) as should_reject,
    ROUND(SUM(CASE WHEN recommendation = 'index' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_ready,
    SUM(CASE WHEN has_pii THEN 1 ELSE 0 END) as pii_violations,
    SUM(CASE WHEN is_duplicate THEN 1 ELSE 0 END) as duplicates,
    SUM(CASE WHEN chunk_quality_flag = 'optimal' THEN 1 ELSE 0 END) as optimal_chunks
FROM read_parquet('data/gold/reviews_featured.parquet');

