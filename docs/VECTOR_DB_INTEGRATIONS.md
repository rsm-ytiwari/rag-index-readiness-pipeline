# Vector Database Integrations

## Overview

Quality-filtered reviews can be uploaded to any vector database for RAG applications.

## Supported Databases

| Database | Cost | Best For | Setup Time |
|----------|------|----------|------------|
| **Pinecone** | $70/mo | Production, managed | 10 min |
| **Weaviate** | Free (self-host) | Full control | 30 min |
| **ChromaDB** | Free | Development | 5 min |
| **Qdrant** | Free (self-host) | High performance | 20 min |
| **Milvus** | Free (self-host) | Large scale | 45 min |

## Quick Start: ChromaDB (Free)
```python
import chromadb
import pandas as pd

# Load quality-filtered reviews
df = pd.read_parquet('data/gold/reviews_featured.parquet')
ready = df[df['index_ready']]

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("reviews")

# Add documents
collection.add(
    documents=ready['text'].tolist(),
    ids=ready['review_id'].tolist(),
    metadatas=[{'stars': float(s)} for s in ready['stars']]
)

# Query
results = collection.query(
    query_texts=["best pizza"],
    n_results=5
)
```

## Production: Pinecone

See `examples/pinecone_integration.py` for complete code.

**Costs:**
- Pinecone: $70/month
- OpenAI embeddings: $0.13/1M tokens
- Total: ~$71/month for 100K reviews

## Self-Hosted: Weaviate
```bash
# Start Weaviate
docker run -p 8080:8080 semitechnologies/weaviate

# Upload reviews
python examples/weaviate_integration.py
```

**Costs:** $0 (self-hosted)

## Best Practices

1. **Always filter first:** Only index reviews with `index_ready = True`
2. **Batch uploads:** Upload in batches of 100-1000
3. **Include metadata:** Add quality_score, stars, date for filtering
4. **Monitor costs:** Track embedding API usage
5. **Test locally:** Use ChromaDB for development

## Query Examples

### Semantic Search
```python
results = collection.query(
    query_texts=["amazing pasta"],
    where={"quality_score": {"$gte": 80}},
    n_results=10
)
```

### Filtered Search
```python
results = collection.query(
    query_texts=["romantic dinner"],
    where={"stars": {"$gte": 4}},
    n_results=5
)
```

## Next Steps

1. Choose vector database
2. Run appropriate integration script
3. Test with sample queries
4. Monitor performance and costs
