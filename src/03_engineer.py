"""
03_engineer.py
Feature Engineering: Chunking + Duplicates + PII

Silver → Gold Parquet with quality features:
1. Chunking features (chunk_count, avg_chunk_tokens, chunk_quality_flag)
2. Duplicate detection (is_duplicate, duplicate_cluster_id)
3. PII detection (has_pii, pii_types)

Input:  data/silver/reviews_cleaned.parquet (98,589 rows)
Output: data/gold/reviews_featured.parquet (98,589 rows with features)
"""

import pandas as pd
from pathlib import Path
import sys
import time
from datasketch import MinHash, MinHashLSH
import json

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import chunk_text, detect_pii, analyze_chunk_quality

print("="*70)
print("03_ENGINEER.PY - Feature Engineering Pipeline")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

INPUT_FILE = SILVER_DIR / "reviews_cleaned.parquet"
OUTPUT_FILE = GOLD_DIR / "reviews_featured.parquet"

# Duplicate detection parameters
DUPLICATE_THRESHOLD = 0.8  # 80% similarity
NUM_PERM = 128  # MinHash permutations (higher = more accurate, slower)

print(f"\n📁 Input:  {INPUT_FILE}")
print(f"📁 Output: {OUTPUT_FILE}")
print(f"🎯 Duplicate threshold: {DUPLICATE_THRESHOLD} (80% similarity)")
print(f"🎯 MinHash permutations: {NUM_PERM}")

# ============================================================================
# STEP 1: Load Silver Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 1: Loading Silver Parquet")
print(f"{'─'*70}")

start_time = time.time()
df = pd.read_parquet(INPUT_FILE)
elapsed = time.time() - start_time

print(f"✅ Loaded {len(df):,} reviews")
print(f"⏱️  Load time: {elapsed:.1f} seconds")
print(f"📊 Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: TEST on 10,000 Reviews First
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 2: Testing on 10,000 Reviews")
print(f"{'─'*70}")

# Take a sample for testing
test_sample = df.head(10000).copy()
print(f"Testing pipeline on {len(test_sample):,} reviews...")

# Test chunking
print("\n  Testing chunking...")
start_time = time.time()
test_sample['chunks'] = test_sample['text'].apply(chunk_text)
elapsed = time.time() - start_time
print(f"  ✅ Chunking complete ({elapsed:.1f}s, {len(test_sample)/elapsed:.0f} reviews/sec)")

# Test chunk analysis
print("  Testing chunk analysis...")
test_sample['chunk_stats'] = test_sample['chunks'].apply(analyze_chunk_quality)
print(f"  ✅ Chunk analysis complete")

# Test PII detection
print("  Testing PII detection...")
start_time = time.time()
pii_results = test_sample['text'].apply(detect_pii)
test_sample['has_pii'] = pii_results.apply(lambda x: x[0])
test_sample['pii_types'] = pii_results.apply(lambda x: x[1])
elapsed = time.time() - start_time
print(f"  ✅ PII detection complete ({elapsed:.1f}s, {len(test_sample)/elapsed:.0f} reviews/sec)")

# Show sample results
print(f"\n  📊 Test Results:")
print(f"    Reviews with chunks: {(test_sample['chunks'].apply(len) > 0).sum():,}")
print(f"    Reviews with PII: {test_sample['has_pii'].sum():,} ({test_sample['has_pii'].sum()/len(test_sample)*100:.1f}%)")
print(f"    Chunk quality distribution:")
quality_counts = test_sample['chunk_stats'].apply(lambda x: x['quality_flag']).value_counts()
for flag, count in quality_counts.items():
    print(f"      {flag}: {count:,} ({count/len(test_sample)*100:.1f}%)")

print(f"\n  ✅ Test successful - proceeding with full dataset")

# ============================================================================
# STEP 3: Apply Chunking to Full Dataset
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 3: Chunking Analysis (Full Dataset)")
print(f"{'─'*70}")

print(f"Applying chunk_text() to {len(df):,} reviews...")
print("This will take ~5-10 minutes...")

start_time = time.time()

# Apply chunking
df['chunks'] = df['text'].apply(chunk_text)

# Progress indicator
elapsed = time.time() - start_time
print(f"✅ Chunking complete")
print(f"⏱️  Time: {elapsed:.1f} seconds ({len(df)/elapsed:.0f} reviews/sec)")

# Analyze chunk quality
print("\nComputing chunk statistics...")
start_time = time.time()

df['chunk_stats'] = df['chunks'].apply(analyze_chunk_quality)

elapsed = time.time() - start_time
print(f"✅ Chunk analysis complete")
print(f"⏱️  Time: {elapsed:.1f} seconds")

# Extract chunk statistics into separate columns
print("\nExtracting chunk features...")
df['chunk_count'] = df['chunk_stats'].apply(lambda x: x['chunk_count'])
df['avg_chunk_tokens'] = df['chunk_stats'].apply(lambda x: x['avg_chunk_tokens'])
df['min_chunk_tokens'] = df['chunk_stats'].apply(lambda x: x['min_chunk_tokens'])
df['max_chunk_tokens'] = df['chunk_stats'].apply(lambda x: x['max_chunk_tokens'])
df['chunk_quality_flag'] = df['chunk_stats'].apply(lambda x: x['quality_flag'])

# Drop intermediate columns
df = df.drop(columns=['chunks', 'chunk_stats'])

print(f"✅ Chunk features extracted")

# ============================================================================
# STEP 4: Validate Chunk Distribution
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 4: Chunk Distribution Validation")
print(f"{'─'*70}")

print(f"\n📊 Chunk Count Distribution:")
print(df['chunk_count'].value_counts().sort_index().head(10))

print(f"\n📊 Chunk Quality Distribution:")
quality_dist = df['chunk_quality_flag'].value_counts()
for flag, count in quality_dist.items():
    pct = count / len(df) * 100
    print(f"  {flag:15s}: {count:6,} ({pct:5.1f}%)")

print(f"\n📊 Average Chunk Tokens Statistics:")
print(df['avg_chunk_tokens'].describe())

# ============================================================================
# STEP 5: Duplicate Detection (MinHash LSH)
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 5: Duplicate Detection (MinHash LSH)")
print(f"{'─'*70}")

print(f"Detecting near-duplicates at {DUPLICATE_THRESHOLD*100:.0f}% similarity threshold...")
print("This will take ~10-15 minutes...")

start_time = time.time()

# Initialize LSH index
lsh = MinHashLSH(threshold=DUPLICATE_THRESHOLD, num_perm=NUM_PERM)

# Store MinHash signatures
minhash_dict = {}

print("\n  Step 5a: Computing MinHash signatures...")
for idx, row in df.iterrows():
    # Create MinHash signature for this review
    m = MinHash(num_perm=NUM_PERM)
    
    # Tokenize text (simple word-based)
    words = row['text'].lower().split()
    for word in words:
        m.update(word.encode('utf-8'))
    
    minhash_dict[row['review_id']] = m
    
    # Progress indicator
    if (idx + 1) % 10000 == 0:
        print(f"    Processed {idx+1:,} reviews...", end='\r')

print(f"\n  ✅ MinHash signatures computed for {len(minhash_dict):,} reviews")

elapsed_minhash = time.time() - start_time
print(f"  ⏱️  Time: {elapsed_minhash:.1f} seconds")

# Build LSH index and find duplicates
print("\n  Step 5b: Building LSH index and finding duplicates...")
start_time = time.time()

duplicate_clusters = {}
cluster_id = 0

for review_id, minhash in minhash_dict.items():
    # Query for similar items
    result = lsh.query(minhash)
    
    if result:
        # Found duplicates - assign to existing cluster
        # Use the first result's cluster
        found_cluster = None
        for existing_id in result:
            if existing_id in duplicate_clusters:
                found_cluster = duplicate_clusters[existing_id]
                break
        
        if found_cluster is not None:
            duplicate_clusters[review_id] = found_cluster
        else:
            # Create new cluster
            duplicate_clusters[review_id] = cluster_id
            for dup_id in result:
                duplicate_clusters[dup_id] = cluster_id
            cluster_id += 1
    else:
        # No duplicates found - this is unique
        duplicate_clusters[review_id] = -1  # -1 means unique
    
    # Insert into LSH
    lsh.insert(review_id, minhash)
    
    # Progress indicator
    if (len(duplicate_clusters) % 10000 == 0):
        print(f"    Processed {len(duplicate_clusters):,} reviews...", end='\r')

elapsed_lsh = time.time() - start_time
total_dup_time = elapsed_minhash + elapsed_lsh

print(f"\n  ✅ LSH index built and duplicates found")
print(f"  ⏱️  Time: {elapsed_lsh:.1f} seconds")
print(f"  ⏱️  Total duplicate detection time: {total_dup_time:.1f} seconds")

# Add duplicate information to dataframe
df['duplicate_cluster_id'] = df['review_id'].map(duplicate_clusters)
df['is_duplicate'] = df['duplicate_cluster_id'] != -1

duplicate_count = df['is_duplicate'].sum()
duplicate_rate = duplicate_count / len(df) * 100

print(f"\n📊 Duplicate Detection Results:")
print(f"  Total reviews:     {len(df):,}")
print(f"  Duplicates found:  {duplicate_count:,} ({duplicate_rate:.1f}%)")
print(f"  Unique reviews:    {len(df) - duplicate_count:,} ({100-duplicate_rate:.1f}%)")
print(f"  Duplicate clusters: {df[df['is_duplicate']]['duplicate_cluster_id'].nunique():,}")

# ============================================================================
# STEP 6: PII Detection
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 6: PII Detection (Full Dataset)")
print(f"{'─'*70}")

print(f"Scanning {len(df):,} reviews for PII...")
print("This will take ~2-3 minutes...")

start_time = time.time()

# Apply PII detection
pii_results = df['text'].apply(detect_pii)
df['has_pii'] = pii_results.apply(lambda x: x[0])
df['pii_types'] = pii_results.apply(lambda x: json.dumps(x[1]))  # Store as JSON string

elapsed = time.time() - start_time
print(f"✅ PII detection complete")
print(f"⏱️  Time: {elapsed:.1f} seconds ({len(df)/elapsed:.0f} reviews/sec)")

pii_count = df['has_pii'].sum()
pii_rate = pii_count / len(df) * 100

print(f"\n📊 PII Detection Results:")
print(f"  Total reviews:       {len(df):,}")
print(f"  Reviews with PII:    {pii_count:,} ({pii_rate:.1f}%)")
print(f"  Reviews without PII: {len(df) - pii_count:,} ({100-pii_rate:.1f}%)")

# Count PII types
print(f"\n📊 PII Types Breakdown:")
pii_type_counts = {}
for pii_list_str in df[df['has_pii']]['pii_types']:
    pii_list = json.loads(pii_list_str)
    for pii_type in pii_list:
        pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1

for pii_type, count in sorted(pii_type_counts.items()):
    print(f"  {pii_type:10s}: {count:,}")

# ============================================================================
# STEP 7: Write to Gold Parquet
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 7: Writing to Gold Parquet")
print(f"{'─'*70}")

# Ensure output directory exists
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# Select columns for output
output_columns = [
    'review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool',
    'text', 'date', 'token_count',
    'chunk_count', 'avg_chunk_tokens', 'min_chunk_tokens', 'max_chunk_tokens', 'chunk_quality_flag',
    'is_duplicate', 'duplicate_cluster_id',
    'has_pii', 'pii_types'
]

df_output = df[output_columns].copy()

print(f"Writing {len(df_output):,} reviews with {len(output_columns)} columns...")

start_time = time.time()

df_output.to_parquet(
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
# STEP 8: Reload and Validate
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 8: Validation")
print(f"{'─'*70}")

start_time = time.time()
df_validate = pd.read_parquet(OUTPUT_FILE)
elapsed = time.time() - start_time

print(f"✅ File reloaded successfully")
print(f"⏱️  Read time: {elapsed:.1f} seconds")

print(f"\n📊 Validation Results:")
print(f"  Shape: {df_validate.shape}")
print(f"  Columns: {df_validate.columns.tolist()}")

# Check for required columns
required_cols = ['chunk_count', 'chunk_quality_flag', 'is_duplicate', 'has_pii']
for col in required_cols:
    assert col in df_validate.columns, f"❌ Missing column: {col}"
    print(f"  ✅ Column '{col}' present")

print(f"\n✅ All validation checks passed")

# ============================================================================
# STEP 9: Feature Engineering Summary
# ============================================================================

print(f"\n{'─'*70}")
print("STEP 9: Feature Engineering Summary")
print(f"{'─'*70}")

print(f"\n📊 Final Dataset Statistics:")
print(f"  Total reviews:          {len(df_validate):,}")
print(f"  Total columns:          {len(df_validate.columns)}")
print(f"  File size:              {file_size_mb:.1f} MB")

print(f"\n📊 Chunk Quality Distribution:")
for flag, count in df_validate['chunk_quality_flag'].value_counts().items():
    pct = count / len(df_validate) * 100
    print(f"  {flag:15s}: {count:6,} ({pct:5.1f}%)")

print(f"\n📊 Duplicate Rate:")
print(f"  Duplicates:   {df_validate['is_duplicate'].sum():,} ({df_validate['is_duplicate'].sum()/len(df_validate)*100:.1f}%)")
print(f"  Unique:       {(~df_validate['is_duplicate']).sum():,} ({(~df_validate['is_duplicate']).sum()/len(df_validate)*100:.1f}%)")

print(f"\n📊 PII Detection:")
print(f"  With PII:     {df_validate['has_pii'].sum():,} ({df_validate['has_pii'].sum()/len(df_validate)*100:.1f}%)")
print(f"  Without PII:  {(~df_validate['has_pii']).sum():,} ({(~df_validate['has_pii']).sum()/len(df_validate)*100:.1f}%)")

print(f"\n📊 Sample Featured Reviews:")
print(df_validate[['review_id', 'stars', 'token_count', 'chunk_count', 'chunk_quality_flag', 'is_duplicate', 'has_pii']].head(5).to_string())

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*70}")
print("✅ 03_ENGINEER.PY COMPLETED SUCCESSFULLY")
print(f"{'='*70}")
print(f"📂 Output: {OUTPUT_FILE}")
print(f"📊 Rows: {len(df_validate):,}")
print(f"📊 Features: {len(output_columns)} columns")
print(f"💾 Size: {file_size_mb:.1f} MB")
print(f"{'='*70}\n")
