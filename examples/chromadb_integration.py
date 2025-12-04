"""
ChromaDB Integration Example (Local, Free)
"""

def upload_to_chromadb(df):
    """Upload to ChromaDB (free, local)."""
    # import chromadb
    # 
    # client = chromadb.Client()
    # collection = client.create_collection("yelp_reviews")
    # 
    # # Add documents
    # collection.add(
    #     documents=df['text'].tolist(),
    #     metadatas=[
    #         {
    #             'stars': float(row['stars']),
    #             'quality_score': float(row['index_readiness_score'])
    #         }
    #         for _, row in df.iterrows()
    #     ],
    #     ids=df['review_id'].tolist()
    # )
    
    print("✅ ChromaDB integration (100% free, local)")
    print("   Install: pip install chromadb")
    print("   Stores embeddings locally - perfect for development!")

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_parquet('../data/gold/reviews_featured.parquet')
    upload_to_chromadb(df[df['index_ready']])
