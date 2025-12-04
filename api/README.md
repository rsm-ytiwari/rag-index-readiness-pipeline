# RAG Index-Readiness API

REST API for real-time review quality scoring before RAG indexing.

## Quick Start
```bash
# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at:
# - http://localhost:8000
# - Interactive docs: http://localhost:8000/docs
```

## Endpoints

### POST /score
Score a single review for index-readiness.

**Request:**
```json
{
  "text": "Amazing food and great service!",
  "stars": 5,
  "review_id": "abc123"
}
```

**Response:**
```json
{
  "review_id": "abc123",
  "index_readiness_score": 100.0,
  "recommendation": "index",
  "chunk_quality": "optimal",
  "has_pii": false,
  "pii_types": [],
  "is_duplicate": false,
  "processing_time_ms": 1.2,
  "metadata": {
    "stars": 5,
    "text_length": 31,
    "chunk_count": 1
  }
}
```

### POST /batch
Score multiple reviews (max 100).

**Request:**
```json
{
  "reviews": [
    {"text": "Great pizza!", "stars": 5},
    {"text": "Terrible service", "stars": 1}
  ]
}
```

**Response:**
```json
{
  "results": [...],
  "total_reviews": 2,
  "avg_score": 85.0,
  "processing_time_ms": 2.5
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-03T16:50:00",
  "uptime_seconds": 120.5
}
```

### GET /stats
API usage statistics.

**Response:**
```json
{
  "total_requests": 42,
  "total_reviews_scored": 156,
  "avg_processing_time_ms": 0.8,
  "started_at": "2025-12-03T16:45:00",
  "uptime_seconds": 300.0
}
```

## Scoring Logic

**Index-Readiness Score (0-100):**
- **40%** Chunk Quality (optimal=100, too_short=50, too_long=0)
- **30%** Duplicate Score (unique=100, duplicate=0)
- **30%** PII Score (no PII=100, has PII=0)

**Recommendations:**
- `score >= 70` → **index** (ready for RAG)
- `score >= 50` → **review** (needs manual check)
- `score < 50` → **reject** (do not index)

## Performance

- **Avg processing time:** <1ms per review
- **Throughput:** ~1,000 reviews/second
- **Batch limit:** 100 reviews per request

## Production Deployment

### Using Docker
```bash
# Build image
docker build -t rag-api .

# Run container
docker run -p 8000:8000 rag-api
```

### Using systemd
```bash
# Create service file: /etc/systemd/system/rag-api.service
[Unit]
Description=RAG Index-Readiness API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/rag-api
ExecStart=/usr/local/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Integration Examples

### Python
```python
import requests

response = requests.post("http://localhost:8000/score", json={
    "text": "Great food!",
    "stars": 5
})
result = response.json()
print(f"Score: {result['index_readiness_score']}")
```

### cURL
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"text":"Great food!","stars":5}'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/score', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Great food!', stars: 5})
});
const result = await response.json();
console.log(`Score: ${result.index_readiness_score}`);
```

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `422` - Validation error (invalid input)
- `500` - Internal server error

**Example validation error:**
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 10 characters"
    }
  ]
}
```

## License

MIT
