"""
Pinecone Vector Database Integration Example

Shows how to upload quality-filtered reviews to Pinecone.
"""

import pandas as pd
from typing import List, Dict
import os

# Note: Install with: pip install pinecone-client openai
# from pinecone import Pinecone, ServerlessSpec
# from openai import OpenAI

def upload_to_pinecone(df: pd.DataFrame, api_key: str = None):
    """
    Upload index-ready reviews to Pinecone.
    
    Args:
        df: DataFrame with quality-filtered reviews
        api_key: Pinecone API key (or set PINECONE_API_KEY env var)
    
    Example:
        >>> df = pd.read_parquet('data/gold/reviews_featured.parquet')
        >>> ready_reviews = df[df['index_ready']]
        >>> upload_to_pinecone(ready_reviews)
    """
    # Initialize Pinecone
    api_key = api_key or os.environ.get('PINECONE_API_KEY')
    
    # Uncomment when ready to use:
    # pc = Pinecone(api_key=api_key)
    # 
    # # Create index (if doesn't exist)
    # index_name = "yelp-reviews"
    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(
    #         name=index_name,
    #         dimension=1536,  # OpenAI text-embedding-3-small
    #         metric='cosine',
    #         spec=ServerlessSpec(cloud='aws', region='us-east-1')
    #     )
    # 
    # index = pc.Index(index_name)
    # 
    # # Initialize OpenAI for embeddings
    # client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    # 
    # # Upload in batches
    # batch_size = 100
    # for i in range(0, len(df), batch_size):
    #     batch = df.iloc[i:i+batch_size]
    #     
    #     # Generate embeddings
    #     texts = batch['text'].tolist()
    #     embeddings = client.embeddings.create(
    #         input=texts,
    #         model="text-embedding-3-small"
    #     )
    #     
    #     # Prepare vectors
    #     vectors = []
    #     for idx, (_, row) in enumerate(batch.iterrows()):
    #         vectors.append({
    #             'id': row['review_id'],
    #             'values': embeddings.data[idx].embedding,
    #             'metadata': {
    #                 'text': row['text'],
    #                 'stars': float(row['stars']),
    #                 'quality_score': float(row['index_readiness_score']),
    #                 'business_id': row['business_id']
    #             }
    #         })
    #     
    #     # Upsert to Pinecone
    #     index.upsert(vectors=vectors)
    #     print(f"Uploaded batch {i//batch_size + 1}: {len(vectors)} vectors")
    
    print(f"""
    ✅ Integration Example: Pinecone
    
    To use this integration:
    1. Install: pip install pinecone-client openai
    2. Get API keys from https://www.pinecone.io and https://platform.openai.com
    3. Set environment variables:
       export PINECONE_API_KEY="your-key"
       export OPENAI_API_KEY="your-key"
    4. Uncomment the code above
    5. Run: python examples/pinecone_integration.py
    
    Cost estimate:
    - Pinecone: $70/month (starter plan)
    - OpenAI embeddings: $0.13 per 1M tokens (~$1.30 for 100K reviews)
    - Total: ~$71.30/month for 100K reviews
    """)

if __name__ == '__main__':
    print("="*70)
    print("PINECONE INTEGRATION EXAMPLE")
    print("="*70)
    
    # Load quality-filtered reviews
    df = pd.read_parquet('../data/gold/reviews_featured.parquet')
    ready_reviews = df[df['index_ready']]
    
    print(f"\n📊 Reviews ready for indexing: {len(ready_reviews):,}")
    print(f"📊 Quality score: {ready_reviews['index_readiness_score'].mean():.1f}/100")
    
    upload_to_pinecone(ready_reviews)
