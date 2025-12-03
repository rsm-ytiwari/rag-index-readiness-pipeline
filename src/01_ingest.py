"""
01_ingest.py
Load 100K sample from Yelp reviews (6.99M total) → Bronze Parquet

Input:  data/Raw/yelp_academic_dataset_review.json (JSONL format)
Output: data/bronze/reviews_raw.parquet (100K rows)
"""

import pandas as pd
import json
from pathlib import Path
import time
import sys

print("="*70)
print("01_INGEST.PY - Data Ingestion Pipeline")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
RAW_DIR = DATA_DIR / "Raw"
INPUT_FILE = RAW_DIR / "yelp_academic_dataset_review.json"

SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

print(f"\n📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Raw Data Dir: {RAW_DIR}")
print(f"📁 Input File:   {INPUT_FILE}")
print(f"📁 Output Dir:   {BRONZE_DIR}")
print(f"🎯 Sample Size:  {SAMPLE_SIZE:,}")
print(f"🔢 Random Seed:  {RANDOM_SEED}")

# ============================================================================
# STEP 1: Verify Input File
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 1: Verifying Input File")
print(f"{'─'*70}")

if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found at {INPUT_FILE}")
    print(f"📂 Checking if directory exists: {RAW_DIR.exists()}")
    if RAW_DIR.exists():
        print(f"📂 Files in Raw directory:")
        for f in RAW_DIR.iterdir():
            print(f"   - {f.name}")
    sys.exit(1)

file_size_gb = INPUT_FILE.stat().st_size / (1024**3)
print(f"✅ File exists: {INPUT_FILE.name}")
print(f"✅ File size: {file_size_gb:.2f} GB")

# ============================================================================
# STEP 2: Count Total Rows (Quick Scan)
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 2: Counting Total Rows")
print(f"{'─'*70}")

start_time = time.time()
total_rows = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        total_rows += 1
        if total_rows % 1_000_000 == 0:
            print(f"  Counted {total_rows:,} rows...", end='\r')

elapsed = time.time() - start_time
print(f"\n✅ Total rows in dataset: {total_rows:,}")
print(f"⏱️  Counting time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 3: Sample Data (Reservoir Sampling)
# ============================================================================

print(f"\n{'─'*70}")
print(f"STEP 3: Sampling {SAMPLE_SIZE:,} Rows")
print(f"{'─'*70}")

import random
random.seed(RANDOM_SEED)

# Reservoir sampling for memory efficiency
reservoir = []
start_time = time.time()

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # First SAMPLE_SIZE items go directly into reservoir
        if i < SAMPLE_SIZE:
            reservoir.append(line)
        else:
            # Random replacement with decreasing probability
            j = random.randint(0, i)
            if j < SAMPLE_SIZE:
                reservoir[j] = line
        
        # Progress indicator
        if (i + 1) % 500_000 == 0:
            print(f"  Processed {i+1:,} rows...", end='\r')

elapsed = time.time() - start_time
print(f"\n✅ Sampling complete: {len(reservoir):,} rows selected")
print(f"⏱️  Sampling time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 4: Parse JSON Lines into DataFrame
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 4: Parsing JSON to DataFrame")
print(f"{'─'*70}")

start_time = time.time()
records = []

for i, line in enumerate(reservoir):
    try:
        record = json.loads(line)
        records.append(record)
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Failed to parse line {i}: {e}")
        continue
    
    if (i + 1) % 10_000 == 0:
        print(f"  Parsed {i+1:,} records...", end='\r')

df = pd.DataFrame(records)
elapsed = time.time() - start_time

print(f"\n✅ DataFrame created")
print(f"⏱️  Parse time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 5: Basic Validation
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 5: Data Validation")
print(f"{'─'*70}")

print(f"\n📊 DataFrame Shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")
print(f"\n📊 Data Types:")
print(df.dtypes)
print(f"\n📊 Null Counts:")
print(df.isnull().sum())
print(f"\n📊 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Critical validation checks
assert len(df) == SAMPLE_SIZE, f"❌ Expected {SAMPLE_SIZE} rows, got {len(df)}"
assert 'text' in df.columns, "❌ Missing 'text' column"
assert 'review_id' in df.columns, "❌ Missing 'review_id' column"
assert 'stars' in df.columns, "❌ Missing 'stars' column"

print(f"\n✅ All validation checks passed")

# ============================================================================
# STEP 6: Write to Bronze Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 6: Writing to Bronze Parquet")
print(f"{'─'*70}")

# Ensure output directory exists
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

output_file = BRONZE_DIR / "reviews_raw.parquet"
start_time = time.time()

df.to_parquet(
    output_file,
    engine='pyarrow',
    compression='snappy',
    index=False
)

elapsed = time.time() - start_time
file_size_mb = output_file.stat().st_size / (1024**2)

print(f"✅ Parquet file written: {output_file.name}")
print(f"✅ File size: {file_size_mb:.1f} MB")
print(f"⏱️  Write time: {elapsed:.1f} seconds")

# ============================================================================
# STEP 7: Reload and Verify
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 7: Reloading and Verifying")
print(f"{'─'*70}")

start_time = time.time()
df_validate = pd.read_parquet(output_file)
elapsed = time.time() - start_time

print(f"✅ File reloaded successfully")
print(f"⏱️  Read time: {elapsed:.1f} seconds")
print(f"📊 Shape: {df_validate.shape}")
print(f"📊 Columns: {df_validate.columns.tolist()}")

# Validate row count matches
assert len(df_validate) == SAMPLE_SIZE, f"❌ Row count mismatch: {len(df_validate)} != {SAMPLE_SIZE}"

print(f"\n✅ Validation successful: {len(df_validate):,} rows")

# ============================================================================
# STEP 8: Display Sample
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 8: Sample Data")
print(f"{'─'*70}")

print(f"\n📝 First 2 reviews:")
print(df_validate[['review_id', 'stars', 'text']].head(2).to_string())

print(f"\n📊 Stars distribution:")
print(df_validate['stars'].value_counts().sort_index())

print(f"\n📊 Text length statistics:")
text_lengths = df_validate['text'].str.len()
print(f"  Min length: {text_lengths.min()}")
print(f"  Max length: {text_lengths.max()}")
print(f"  Mean length: {text_lengths.mean():.1f}")
print(f"  Median length: {text_lengths.median():.1f}")

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*70}")
print("✅ 01_INGEST.PY COMPLETED SUCCESSFULLY")
print(f"{'='*70}")
print(f"📂 Output: {output_file}")
print(f"📊 Rows: {len(df_validate):,}")
print(f"📊 Columns: {len(df_validate.columns)}")
print(f"💾 Size: {file_size_mb:.1f} MB")
print(f"{'='*70}\n")
