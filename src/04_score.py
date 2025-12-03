"""
04_score.py
Add index-readiness scoring to Gold Parquet

Scoring formula:
- Chunk quality: 40% weight (100 if optimal, 50 if borderline, 0 if poor)
- Duplicate penalty: 30% weight (0 if duplicate, 100 if unique)
- PII penalty: 30% weight (0 if has PII, 100 if clean)

Final score: 0-100
- Score >= 70: Index Ready (use in RAG system)
- Score 50-69: Needs Review (manual check)
- Score < 50: Reject (do not index)

Input:  data/gold/reviews_featured.parquet
Output: data/gold/reviews_featured.parquet (updated with scores)
"""

import pandas as pd
from pathlib import Path
import time

print("="*70)
print("04_SCORE.PY - Index-Readiness Scoring")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"

INPUT_FILE = GOLD_DIR / "reviews_featured.parquet"
OUTPUT_FILE = GOLD_DIR / "reviews_featured.parquet"  # Overwrite with scores

# Scoring thresholds
READY_THRESHOLD = 70  # Score >= 70 is "index ready"
REVIEW_THRESHOLD = 50  # Score 50-69 needs manual review

print(f"\n�� Input/Output: {INPUT_FILE}")
print(f"🎯 Index ready threshold: {READY_THRESHOLD}")
print(f"🎯 Needs review threshold: {REVIEW_THRESHOLD}")

# ============================================================================
# STEP 1: Load Gold Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 1: Loading Gold Parquet")
print(f"{'─'*70}")

start_time = time.time()
df = pd.read_parquet(INPUT_FILE)
elapsed = time.time() - start_time

print(f"✅ Loaded {len(df):,} reviews")
print(f"⏱️  Load time: {elapsed:.1f} seconds")
print(f"📊 Columns: {len(df.columns)}")

# ============================================================================
# STEP 2: Implement Scoring Formula
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 2: Computing Index-Readiness Scores")
print(f"{'─'*70}")

print("\nScoring weights:")
print("  - Chunk quality:    40%")
print("  - Duplicate status: 30%")
print("  - PII status:       30%")

start_time = time.time()

# -------------------------------------------------------------------------
# Component 1: Chunk Quality Score (0-100)
# -------------------------------------------------------------------------

def compute_chunk_score(quality_flag: str) -> int:
    """
    Score based on chunk quality.
    
    optimal:    100 (perfect for RAG)
    too_short:   50 (usable but not ideal)
    too_long:     0 (needs re-chunking)
    empty:        0 (invalid)
    """
    if quality_flag == 'optimal':
        return 100
    elif quality_flag == 'too_short':
        return 50
    else:  # too_long, empty, or any other
        return 0

df['chunk_score'] = df['chunk_quality_flag'].apply(compute_chunk_score)

print(f"\n✅ Chunk scores computed")
print(f"   Distribution:")
print(f"     Score 100 (optimal):   {(df['chunk_score'] == 100).sum():,} ({(df['chunk_score'] == 100).sum()/len(df)*100:.1f}%)")
print(f"     Score 50 (too_short):  {(df['chunk_score'] == 50).sum():,} ({(df['chunk_score'] == 50).sum()/len(df)*100:.1f}%)")
print(f"     Score 0 (poor):        {(df['chunk_score'] == 0).sum():,} ({(df['chunk_score'] == 0).sum()/len(df)*100:.1f}%)")

# -------------------------------------------------------------------------
# Component 2: Duplicate Score (0 or 100)
# -------------------------------------------------------------------------

def compute_duplicate_score(is_duplicate: bool) -> int:
    """
    Score based on duplicate status.
    
    False (unique):    100 (good for indexing)
    True (duplicate):    0 (skip - wastes storage)
    """
    return 0 if is_duplicate else 100

df['duplicate_score'] = df['is_duplicate'].apply(compute_duplicate_score)

print(f"\n✅ Duplicate scores computed")
print(f"   Distribution:")
print(f"     Score 100 (unique):     {(df['duplicate_score'] == 100).sum():,} ({(df['duplicate_score'] == 100).sum()/len(df)*100:.1f}%)")
print(f"     Score 0 (duplicate):    {(df['duplicate_score'] == 0).sum():,} ({(df['duplicate_score'] == 0).sum()/len(df)*100:.1f}%)")

# -------------------------------------------------------------------------
# Component 3: PII Score (0 or 100)
# -------------------------------------------------------------------------

def compute_pii_score(has_pii: bool) -> int:
    """
    Score based on PII presence.
    
    False (no PII):  100 (safe to index)
    True (has PII):    0 (privacy risk - manual review needed)
    """
    return 0 if has_pii else 100

df['pii_score'] = df['has_pii'].apply(compute_pii_score)

print(f"\n✅ PII scores computed")
print(f"   Distribution:")
print(f"     Score 100 (no PII):     {(df['pii_score'] == 100).sum():,} ({(df['pii_score'] == 100).sum()/len(df)*100:.1f}%)")
print(f"     Score 0 (has PII):      {(df['pii_score'] == 0).sum():,} ({(df['pii_score'] == 0).sum()/len(df)*100:.1f}%)")

# -------------------------------------------------------------------------
# Final Score: Weighted Average
# -------------------------------------------------------------------------

print(f"\n{'─'*70}")
print("Computing final index-readiness scores...")
print(f"{'─'*70}")

df['index_readiness_score'] = (
    df['chunk_score'] * 0.4 + 
    df['duplicate_score'] * 0.3 + 
    df['pii_score'] * 0.3
).round(1)

elapsed = time.time() - start_time
print(f"✅ All scores computed")
print(f"⏱️  Time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 3: Add index_ready Flag
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 3: Adding Index-Ready Flags")
print(f"{'─'*70}")

# Add binary flag
df['index_ready'] = df['index_readiness_score'] >= READY_THRESHOLD

# Add categorical recommendation
def get_recommendation(score: float) -> str:
    """Categorize reviews by score."""
    if score >= READY_THRESHOLD:
        return 'index'
    elif score >= REVIEW_THRESHOLD:
        return 'review'
    else:
        return 'reject'

df['recommendation'] = df['index_readiness_score'].apply(get_recommendation)

print(f"✅ Flags added")
print(f"   - index_ready: {df['index_ready'].dtype}")
print(f"   - recommendation: {df['recommendation'].dtype}")

# ============================================================================
# STEP 4: Summary Statistics
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 4: Summary Statistics")
print(f"{'─'*70}")

print(f"\n📊 Index-Readiness Score Distribution:")
print(df['index_readiness_score'].describe())

print(f"\n📊 Score Histogram:")
score_bins = [0, 30, 50, 70, 85, 100]
score_labels = ['0-30', '30-50', '50-70', '70-85', '85-100']
df['score_bin'] = pd.cut(df['index_readiness_score'], bins=score_bins, labels=score_labels, include_lowest=True)

for bin_label in score_labels:
    count = (df['score_bin'] == bin_label).sum()
    pct = count / len(df) * 100
    print(f"   {bin_label:8s}: {count:6,} ({pct:5.1f}%)")

print(f"\n📊 Recommendation Distribution:")
rec_counts = df['recommendation'].value_counts()
for rec, count in rec_counts.items():
    pct = count / len(df) * 100
    print(f"   {rec.capitalize():8s}: {count:6,} ({pct:5.1f}%)")

print(f"\n📊 Index-Ready Summary:")
ready_count = df['index_ready'].sum()
not_ready_count = (~df['index_ready']).sum()
print(f"   Ready to index:   {ready_count:6,} ({ready_count/len(df)*100:5.1f}%)")
print(f"   Not ready:        {not_ready_count:6,} ({not_ready_count/len(df)*100:5.1f}%)")

# ============================================================================
# STEP 5: Detailed Breakdown by Quality Issues
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 5: Quality Issues Breakdown")
print(f"{'─'*70}")

print(f"\n📊 Reviews NOT Ready (score < {READY_THRESHOLD}):")
not_ready = df[~df['index_ready']]

print(f"\n   By primary issue:")
issue_breakdown = {
    'Poor chunks only': ((not_ready['chunk_score'] <= 50) & (not_ready['duplicate_score'] == 100) & (not_ready['pii_score'] == 100)).sum(),
    'Has PII only': ((not_ready['chunk_score'] == 100) & (not_ready['duplicate_score'] == 100) & (not_ready['pii_score'] == 0)).sum(),
    'Duplicate only': ((not_ready['chunk_score'] == 100) & (not_ready['duplicate_score'] == 0) & (not_ready['pii_score'] == 100)).sum(),
    'Multiple issues': ((not_ready['chunk_score'] <= 50) | (not_ready['pii_score'] == 0) | (not_ready['duplicate_score'] == 0)).sum()
}

for issue, count in issue_breakdown.items():
    if count > 0:
        pct = count / len(not_ready) * 100 if len(not_ready) > 0 else 0
        print(f"     {issue:20s}: {count:6,} ({pct:5.1f}% of not-ready)")

# ============================================================================
# STEP 6: Sample Reviews by Score Tier
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 6: Sample Reviews by Score Tier")
print(f"{'─'*70}")

print(f"\n📝 HIGH SCORE (index ready, score >= 70):")
high_score = df[df['index_readiness_score'] >= 70].head(2)
for idx, row in high_score.iterrows():
    print(f"   Review {row['review_id']}: Score {row['index_readiness_score']}")
    print(f"     Stars: {row['stars']} | Tokens: {row['token_count']} | Chunk: {row['chunk_quality_flag']}")
    print(f"     Duplicate: {row['is_duplicate']} | PII: {row['has_pii']}")
    print(f"     Text: {row['text'][:100]}...")
    print()

print(f"\n📝 MEDIUM SCORE (needs review, 50-69):")
medium_score = df[(df['index_readiness_score'] >= 50) & (df['index_readiness_score'] < 70)].head(2)
if len(medium_score) > 0:
    for idx, row in medium_score.iterrows():
        print(f"   Review {row['review_id']}: Score {row['index_readiness_score']}")
        print(f"     Stars: {row['stars']} | Tokens: {row['token_count']} | Chunk: {row['chunk_quality_flag']}")
        print(f"     Duplicate: {row['is_duplicate']} | PII: {row['has_pii']}")
        print(f"     Text: {row['text'][:100]}...")
        print()
else:
    print("   (No reviews in this tier)")

print(f"\n📝 LOW SCORE (reject, < 50):")
low_score = df[df['index_readiness_score'] < 50].head(2)
if len(low_score) > 0:
    for idx, row in low_score.iterrows():
        print(f"   Review {row['review_id']}: Score {row['index_readiness_score']}")
        print(f"     Stars: {row['stars']} | Tokens: {row['token_count']} | Chunk: {row['chunk_quality_flag']}")
        print(f"     Duplicate: {row['is_duplicate']} | PII: {row['has_pii']}")
        print(f"     Text: {row['text'][:100]}...")
        print()
else:
    print("   (No reviews in this tier)")

# ============================================================================
# STEP 7: Write Updated Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 7: Writing Updated Gold Parquet")
print(f"{'─'*70}")

# Drop temporary columns
df = df.drop(columns=['score_bin'])

# Reorder columns for better readability
column_order = [
    # IDs and metadata
    'review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'date',
    # Content
    'text', 'token_count',
    # Chunk features
    'chunk_count', 'avg_chunk_tokens', 'min_chunk_tokens', 'max_chunk_tokens', 'chunk_quality_flag',
    # Duplicate features
    'is_duplicate', 'duplicate_cluster_id',
    # PII features
    'has_pii', 'pii_types',
    # Scores (NEW)
    'chunk_score', 'duplicate_score', 'pii_score', 'index_readiness_score', 'index_ready', 'recommendation'
]

df_output = df[column_order].copy()

print(f"Writing {len(df_output):,} reviews with {len(df_output.columns)} columns...")

start_time = time.time()

df_output.to_parquet(
    OUTPUT_FILE,
    engine='pyarrow',
    compression='snappy',
    index=False
)

elapsed = time.time() - start_time
file_size_mb = OUTPUT_FILE.stat().st_size / (1024**2)

print(f"✅ Parquet file updated: {OUTPUT_FILE.name}")
print(f"✅ File size: {file_size_mb:.1f} MB")
print(f"⏱️  Write time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 8: Final Validation
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 8: Final Validation")
print(f"{'─'*70}")

df_validate = pd.read_parquet(OUTPUT_FILE)

print(f"✅ File reloaded successfully")
print(f"📊 Shape: {df_validate.shape}")
print(f"📊 Columns: {len(df_validate.columns)}")

# Check for required columns
required_cols = ['index_readiness_score', 'index_ready', 'recommendation']
for col in required_cols:
    assert col in df_validate.columns, f"❌ Missing column: {col}"
    print(f"  ✅ Column '{col}' present")

print(f"\n✅ All validation checks passed")

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*70}")
print("✅ 04_SCORE.PY COMPLETED SUCCESSFULLY")
print(f"{'='*70}")
print(f"📂 Output: {OUTPUT_FILE}")
print(f"📊 Rows: {len(df_validate):,}")
print(f"📊 Columns: {len(df_validate.columns)} (added 6 score columns)")
print(f"💾 Size: {file_size_mb:.1f} MB")
print(f"\n📊 KEY FINDINGS:")
print(f"   Index-ready:  {(df_validate['index_ready']).sum():,} ({(df_validate['index_ready']).sum()/len(df_validate)*100:.1f}%)")
print(f"   Needs review: {((df_validate['index_readiness_score'] >= 50) & (df_validate['index_readiness_score'] < 70)).sum():,}")
print(f"   Reject:       {(df_validate['index_readiness_score'] < 50).sum():,}")
print(f"{'='*70}\n")
