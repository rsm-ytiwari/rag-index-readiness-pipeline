#!/usr/bin/env python3
"""
Batch Processing Script for RAG Index-Readiness Pipeline
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from utils import chunk_text, detect_pii, analyze_chunk_quality

def setup_logging(log_file=None, log_level="INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    if log_file:
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function} | {message}", level=log_level, rotation="10 MB")

def load_reviews(input_path):
    logger.info(f"Loading reviews from: {input_path}")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        df = pd.read_json(input_path, lines=True)
        logger.success(f"Loaded {len(df):,} reviews (JSONL format)")
    except ValueError:
        df = pd.read_json(input_path)
        logger.success(f"Loaded {len(df):,} reviews (JSON array format)")
    return df

def clean_data(df):
    logger.info("Cleaning data...")
    initial_count = len(df)
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']
    logger.info(f"Removed {initial_count - len(df):,} null/empty reviews")
    
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    df['token_count'] = df['text'].apply(lambda x: len(enc.encode(str(x))))
    
    before_filter = len(df)
    df = df[df['token_count'] >= 20]
    logger.info(f"Removed {before_filter - len(df):,} short reviews (<20 tokens)")
    logger.success(f"Kept {len(df):,} reviews after cleaning")
    return df

def engineer_features(df):
    logger.info("Engineering features...")
    
    # Chunking analysis
    logger.info("Analyzing chunk quality...")
    chunk_results = []
    for idx, row in df.iterrows():
        chunks = chunk_text(row['text'])
        analysis = analyze_chunk_quality(chunks)
        chunk_results.append(analysis)
    
    # Convert to DataFrame
    chunk_df = pd.DataFrame(chunk_results)
    
    # Map quality_flag to chunk_quality_flag (fix the column name mismatch)
    if 'quality_flag' in chunk_df.columns:
        chunk_df['chunk_quality_flag'] = chunk_df['quality_flag']
        chunk_df = chunk_df.drop('quality_flag', axis=1)
    
    # Add all chunk columns to main dataframe
    for col in chunk_df.columns:
        df[col] = chunk_df[col].values
    
    optimal_count = (df['chunk_quality_flag'] == 'optimal').sum()
    logger.success(f"Chunk analysis complete: {optimal_count:,} optimal chunks")
    
    # Duplicate detection
    logger.info("Detecting duplicates...")
    df['text_hash'] = df['text'].apply(lambda x: hash(str(x)))
    df['is_duplicate'] = df.duplicated(subset=['text_hash'], keep='first')
    df['duplicate_cluster_id'] = df.groupby('text_hash').ngroup()
    df.loc[~df['is_duplicate'], 'duplicate_cluster_id'] = -1
    duplicate_count = df['is_duplicate'].sum()
    logger.success(f"Found {duplicate_count:,} duplicates ({duplicate_count/len(df)*100:.1f}%)")
    
    # PII detection
    logger.info("Detecting PII...")
    pii_results = df['text'].apply(detect_pii)
    df['has_pii'] = pii_results.apply(lambda x: x[0])
    df['pii_types'] = pii_results.apply(lambda x: json.dumps(x[1]))
    pii_count = df['has_pii'].sum()
    logger.success(f"Found PII in {pii_count:,} reviews ({pii_count/len(df)*100:.1f}%)")
    
    return df

def score_reviews(df):
    logger.info("Computing index-readiness scores...")
    df['chunk_score'] = df['chunk_quality_flag'].map({
        'optimal': 100,
        'too_short': 50,
        'too_long': 0,
        'empty': 0
    }).fillna(50)
    df['duplicate_score'] = (~df['is_duplicate']).astype(int) * 100
    df['pii_score'] = (~df['has_pii']).astype(int) * 100
    df['index_readiness_score'] = (
        df['chunk_score'] * 0.4 +
        df['duplicate_score'] * 0.3 +
        df['pii_score'] * 0.3
    )
    df['index_ready'] = df['index_readiness_score'] >= 70
    df['recommendation'] = 'reject'
    df.loc[df['index_readiness_score'] >= 50, 'recommendation'] = 'review'
    df.loc[df['index_readiness_score'] >= 70, 'recommendation'] = 'index'
    logger.success(f"Scoring complete. Avg score: {df['index_readiness_score'].mean():.1f}/100")
    return df

def process_batch(input_path, output_path, min_score=70, log_file=None):
    start_time = time.time()
    try:
        df = load_reviews(input_path)
        df = clean_data(df)
        df = engineer_features(df)
        df = score_reviews(df)
        
        logger.info(f"Filtering reviews with score >= {min_score}...")
        df_ready = df[df['index_readiness_score'] >= min_score].copy()
        logger.info(f"Filtered: {len(df_ready):,} / {len(df):,} reviews pass threshold ({len(df_ready)/len(df)*100:.1f}%)")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_ready.to_parquet(output_path, index=False)
        logger.success(f"Saved {len(df_ready):,} ready-to-index reviews to: {output_path}")
        
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*70)
        logger.info(f"Input file: {input_path}")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Total reviews processed: {len(df):,}")
        logger.info(f"Reviews passing threshold (>= {min_score}): {len(df_ready):,} ({len(df_ready)/len(df)*100:.1f}%)")
        logger.info(f"Reviews filtered out: {len(df) - len(df_ready):,} ({(len(df) - len(df_ready))/len(df)*100:.1f}%)")
        logger.info(f"Average quality score: {df['index_readiness_score'].mean():.1f}/100")
        logger.info(f"Processing time: {elapsed:.1f} seconds ({len(df)/elapsed:.0f} reviews/sec)")
        logger.info("="*70)
        return df_ready
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Batch process reviews through RAG quality pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_process.py --input reviews.json --output ready.parquet
    python batch_process.py --input reviews.json --output ready.parquet --min-score 80
    python batch_process.py --input reviews.json --output ready.parquet --log-level DEBUG
        """
    )
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output Parquet file')
    parser.add_argument('--min-score', type=int, default=70, help='Minimum score threshold (default: 70)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', default=None, help='Optional log file path')
    args = parser.parse_args()
    
    setup_logging(log_file=args.log_file, log_level=args.log_level)
    logger.info("="*70)
    logger.info("RAG INDEX-READINESS BATCH PROCESSOR")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    process_batch(input_path=args.input, output_path=args.output, min_score=args.min_score, log_file=args.log_file)
    logger.success("Batch processing complete!")

if __name__ == '__main__':
    main()
