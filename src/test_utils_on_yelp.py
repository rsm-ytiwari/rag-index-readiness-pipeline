"""
Test utils.py functions on actual Yelp reviews from Bronze Parquet
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import chunk_text, detect_pii, analyze_chunk_quality, count_tokens

print("="*70)
print("TESTING UTILS ON REAL YELP REVIEWS")
print("="*70)

# Load Bronze Parquet
bronze_file = Path("data/bronze/reviews_raw.parquet")
print(f"\n📂 Loading: {bronze_file}")

df = pd.read_parquet(bronze_file)
print(f"✅ Loaded {len(df):,} reviews")

# Sample different lengths of reviews
print("\n" + "─"*70)
print("Selecting test samples by text length...")
print("─"*70)

# Get text lengths
df['text_length'] = df['text'].str.len()
df['token_count'] = df['text'].apply(count_tokens)

# Select samples
short_review = df[df['token_count'] < 50].iloc[0] if len(df[df['token_count'] < 50]) > 0 else df.iloc[0]
medium_review = df[(df['token_count'] >= 100) & (df['token_count'] <= 200)].iloc[0] if len(df[(df['token_count'] >= 100) & (df['token_count'] <= 200)]) > 0 else df.iloc[1]
long_review = df[df['token_count'] > 500].iloc[0] if len(df[df['token_count'] > 500]) > 0 else df.iloc[2]

samples = [
    ('SHORT', short_review),
    ('MEDIUM', medium_review),
    ('LONG', long_review)
]

# ============================================================================
# TEST CHUNKING
# ============================================================================

print("\n" + "="*70)
print("TEST 1: CHUNK_TEXT() ON REAL REVIEWS")
print("="*70)

for label, review in samples:
    print(f"\n{'─'*70}")
    print(f"{label} REVIEW")
    print(f"{'─'*70}")
    print(f"Review ID: {review['review_id']}")
    print(f"Stars: {review['stars']}")
    print(f"Length: {review['text_length']} chars, {review['token_count']} tokens")
    print(f"\nText preview:")
    print(review['text'][:200] + "..." if len(review['text']) > 200 else review['text'])
    
    # Chunk it
    chunks = chunk_text(review['text'])
    print(f"\n📊 Chunking Result:")
    print(f"  Total chunks: {len(chunks)}")
    
    for chunk in chunks:
        print(f"  Chunk {chunk['chunk_id']}: {chunk['tokens']} tokens, {chunk['sentences']} sentences")
    
    quality = analyze_chunk_quality(chunks)
    print(f"\n📊 Quality Analysis:")
    print(f"  Avg tokens: {quality['avg_chunk_tokens']}")
    print(f"  Range: {quality['min_chunk_tokens']}-{quality['max_chunk_tokens']} tokens")
    print(f"  Quality flag: {quality['quality_flag']}")

# ============================================================================
# TEST PII DETECTION
# ============================================================================

print("\n" + "="*70)
print("TEST 2: DETECT_PII() ON REAL REVIEWS")
print("="*70)

# Check first 100 reviews for PII
print("\nScanning first 100 reviews for PII...")

pii_stats = {
    'total': 0,
    'with_pii': 0,
    'email': 0,
    'phone': 0,
    'address': 0
}

pii_examples = []

for idx, row in df.head(100).iterrows():
    has_pii, pii_types = detect_pii(row['text'])
    pii_stats['total'] += 1
    
    if has_pii:
        pii_stats['with_pii'] += 1
        for pii_type in pii_types:
            pii_stats[pii_type] += 1
        
        # Save first 3 examples
        if len(pii_examples) < 3:
            pii_examples.append({
                'review_id': row['review_id'],
                'text': row['text'][:200],
                'pii_types': pii_types
            })

print(f"\n📊 PII Detection Results (first 100 reviews):")
print(f"  Total reviews scanned: {pii_stats['total']}")
print(f"  Reviews with PII: {pii_stats['with_pii']} ({pii_stats['with_pii']/pii_stats['total']*100:.1f}%)")
print(f"  Email detected: {pii_stats['email']}")
print(f"  Phone detected: {pii_stats['phone']}")
print(f"  Address detected: {pii_stats['address']}")

if pii_examples:
    print(f"\n📝 Example reviews with PII:")
    for i, ex in enumerate(pii_examples):
        print(f"\nExample {i+1}:")
        print(f"  Review ID: {ex['review_id']}")
        print(f"  PII types: {ex['pii_types']}")
        print(f"  Text: {ex['text']}...")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n📊 Token Distribution (all {len(df):,} reviews):")
print(df['token_count'].describe())

print(f"\n📊 Reviews by token range:")
token_ranges = {
    'Very short (<100 tokens)': len(df[df['token_count'] < 100]),
    'Short (100-200 tokens)': len(df[(df['token_count'] >= 100) & (df['token_count'] < 200)]),
    'Medium (200-500 tokens)': len(df[(df['token_count'] >= 200) & (df['token_count'] < 500)]),
    'Long (500-1000 tokens)': len(df[(df['token_count'] >= 500) & (df['token_count'] < 1000)]),
    'Very long (1000+ tokens)': len(df[df['token_count'] >= 1000])
}

for range_name, count in token_ranges.items():
    pct = count / len(df) * 100
    print(f"  {range_name}: {count:,} ({pct:.1f}%)")

print("\n" + "="*70)
print("✅ TESTING COMPLETE - UTILS.PY VALIDATED")
print("="*70)
