"""
02_clean.py
Clean raw data from Bronze → Silver Parquet

Cleaning steps:
1. Remove rows with null/empty text
2. Remove reviews with <20 tokens (too short)
3. Add token_count column
4. Filter non-English reviews (ASCII heuristic)
5. Write to Silver Parquet

Input:  data/bronze/reviews_raw.parquet (100K rows)
Output: data/silver/reviews_cleaned.parquet (~95K rows)
"""

import pandas as pd
from pathlib import Path
import sys
import time

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import count_tokens

print("="*70)
print("02_CLEAN.PY - Data Cleaning Pipeline")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

INPUT_FILE = BRONZE_DIR / "reviews_raw.parquet"
OUTPUT_FILE = SILVER_DIR / "reviews_cleaned.parquet"

MIN_TOKENS = 20  # Minimum tokens for meaningful analysis

print(f"\n📁 Input:  {INPUT_FILE}")
print(f"📁 Output: {OUTPUT_FILE}")
print(f"🎯 Min tokens: {MIN_TOKENS}")

# ============================================================================
# STEP 1: Load Bronze Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 1: Loading Bronze Parquet")
print(f"{'─'*70}")

start_time = time.time()
df = pd.read_parquet(INPUT_FILE)
elapsed = time.time() - start_time

print(f"✅ Loaded {len(df):,} reviews")
print(f"⏱️  Load time: {elapsed:.1f} seconds")
print(f"\n📊 Columns: {df.columns.tolist()}")

initial_count = len(df)

# ============================================================================
# STEP 2: Remove Rows with Null/Empty Text
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 2: Removing Null/Empty Text")
print(f"{'─'*70}")

# Check for nulls
null_count = df['text'].isnull().sum()
print(f"Null text values: {null_count}")

# Remove nulls
if null_count > 0:
    df = df[df['text'].notnull()].copy()
    print(f"✅ Removed {null_count} null rows")

# Check for empty strings (after stripping whitespace)
df['text'] = df['text'].astype(str).str.strip()
empty_count = (df['text'] == '').sum()
print(f"Empty text values: {empty_count}")

# Remove empty strings
if empty_count > 0:
    df = df[df['text'] != ''].copy()
    print(f"✅ Removed {empty_count} empty rows")

after_null_removal = len(df)
print(f"\n📊 Remaining: {len(df):,} reviews ({len(df)/initial_count*100:.1f}%)")

# ============================================================================
# STEP 3: Add token_count Column
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 3: Adding Token Count Column")
print(f"{'─'*70}")

start_time = time.time()

# Add token count using our utility function
print("Computing token counts...")
df['token_count'] = df['text'].apply(count_tokens)

elapsed = time.time() - start_time
print(f"✅ Token counts computed for {len(df):,} reviews")
print(f"⏱️  Compute time: {elapsed:.1f} seconds")

print(f"\n📊 Token count statistics:")
print(df['token_count'].describe())

# ============================================================================
# STEP 4: Remove Reviews with <20 Tokens
# ============================================================================

print(f"\n{'─'*70}")
print(f"STEP 4: Removing Reviews with <{MIN_TOKENS} Tokens")
print(f"{'─'*70}")

short_reviews = (df['token_count'] < MIN_TOKENS).sum()
print(f"Reviews with <{MIN_TOKENS} tokens: {short_reviews:,} ({short_reviews/len(df)*100:.1f}%)")

# Filter out short reviews
df = df[df['token_count'] >= MIN_TOKENS].copy()

after_token_filter = len(df)
print(f"✅ Removed {short_reviews:,} short reviews")
print(f"\n📊 Remaining: {len(df):,} reviews ({len(df)/initial_count*100:.1f}%)")

# ============================================================================
# STEP 5: Filter Non-English (ASCII Heuristic)
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 5: Filtering Non-English Reviews")
print(f"{'─'*70}")

def is_mostly_ascii(text: str, threshold: float = 0.9) -> bool:
    """
    Check if text is mostly ASCII characters.
    
    Heuristic for English text: >90% ASCII characters.
    This is a simple approximation - more sophisticated NER
    models would be better for production.
    """
    if not text:
        return False
    
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold

# Test on sample first
print("Testing ASCII filter on sample...")
sample = df.head(1000)
non_english_sample = sum(1 for text in sample['text'] if not is_mostly_ascii(text))
print(f"  Sample: {non_english_sample}/1000 ({non_english_sample/10:.1f}%) appear non-English")

# Apply filter
print("\nApplying filter to all reviews...")
start_time = time.time()

df['is_english'] = df['text'].apply(is_mostly_ascii)
non_english_count = (~df['is_english']).sum()

print(f"Non-English reviews: {non_english_count:,} ({non_english_count/len(df)*100:.1f}%)")

# Filter out non-English
df = df[df['is_english']].copy()
df = df.drop(columns=['is_english'])  # Remove temporary column

elapsed = time.time() - start_time
after_english_filter = len(df)

print(f"✅ Removed {non_english_count:,} non-English reviews")
print(f"⏱️  Filter time: {elapsed:.1f} seconds")
print(f"\n📊 Remaining: {len(df):,} reviews ({len(df)/initial_count*100:.1f}%)")

# ============================================================================
# STEP 6: Write to Silver Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 6: Writing to Silver Parquet")
print(f"{'─'*70}")

# Ensure output directory exists
SILVER_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.time()

df.to_parquet(
    OUTPUT_FILE,
    engine='pyarrow',
    compression='snappy',
    index=False
)

elapsed = time.time() - start_time
file_size_mb = OUTPUT_FILE.stat().st_size / (1024**2)

print(f"✅ Parquet file written: {OUTPUT_FILE.name}")
print(f"✅ File size: {file_size_mb:.1f} MB")
print(f"⏱️  Write time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 7: Reload and Validate
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 7: Validation")
print(f"{'─'*70}")

start_time = time.time()
df_validate = pd.read_parquet(OUTPUT_FILE)
elapsed = time.time() - start_time

print(f"✅ File reloaded successfully")
print(f"⏱️  Read time: {elapsed:.1f} seconds")

# Validation checks
print(f"\n📊 Validation Results:")
print(f"  Shape: {df_validate.shape}")
print(f"  Columns: {df_validate.columns.tolist()}")

# Check for nulls
print(f"\n📊 Null counts:")
print(df_validate.isnull().sum())

# Check token count column exists and is correct
assert 'token_count' in df_validate.columns, "❌ Missing token_count column"
assert df_validate['token_count'].min() >= MIN_TOKENS, f"❌ Found reviews with <{MIN_TOKENS} tokens"
assert df_validate['text'].isnull().sum() == 0, "❌ Found null text values"

print(f"\n✅ All validation checks passed")

# ============================================================================
# STEP 8: Cleaning Summary
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 8: Cleaning Summary")
print(f"{'─'*70}")

total_removed = initial_count - len(df_validate)
removal_rate = (total_removed / initial_count) * 100

print(f"\n📊 Cleaning Statistics:")
print(f"  Initial reviews:        {initial_count:,}")
print(f"  After null removal:     {after_null_removal:,} (removed: {initial_count - after_null_removal:,})")
print(f"  After token filter:     {after_token_filter:,} (removed: {after_null_removal - after_token_filter:,})")
print(f"  After English filter:   {after_english_filter:,} (removed: {after_token_filter - after_english_filter:,})")
print(f"  Final count:            {len(df_validate):,}")
print(f"  Total removed:          {total_removed:,} ({removal_rate:.1f}%)")
print(f"  Retention rate:         {len(df_validate)/initial_count*100:.1f}%")

print(f"\n📊 Token distribution (cleaned data):")
print(df_validate['token_count'].describe())

print(f"\n📊 Sample cleaned reviews:")
print(df_validate[['review_id', 'stars', 'token_count', 'text']].head(3).to_string())

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*70}")
print("✅ 02_CLEAN.PY COMPLETED SUCCESSFULLY")
print(f"{'='*70}")
print(f"📂 Output: {OUTPUT_FILE}")
print(f"📊 Rows: {len(df_validate):,}")
print(f"📊 Columns: {len(df_validate.columns)}")
print(f"💾 Size: {file_size_mb:.1f} MB")
print(f"📉 Removed: {removal_rate:.1f}% of original data")
print(f"{'='*70}\n")
