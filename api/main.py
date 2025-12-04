"""
RAG Index-Readiness API
FastAPI REST API for real-time review quality scoring

Endpoints:
- POST /score       - Score a single review
- POST /batch       - Score multiple reviews
- GET /health       - Health check
- GET /stats        - Pipeline statistics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import sys
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import chunk_text, detect_pii, analyze_chunk_quality

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="RAG Index-Readiness API",
    description="Real-time quality scoring for reviews before RAG indexing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global statistics
stats = {
    "total_requests": 0,
    "total_reviews_scored": 0,
    "avg_processing_time_ms": 0,
    "started_at": datetime.now().isoformat()
}

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ReviewInput(BaseModel):
    """Input schema for a single review."""
    text: str = Field(..., min_length=10, max_length=10000, description="Review text content")
    stars: Optional[float] = Field(None, ge=1, le=5, description="Star rating (1-5)")
    review_id: Optional[str] = Field(None, description="Optional review identifier")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Review text cannot be empty')
        return v.strip()

class ReviewOutput(BaseModel):
    """Output schema for scored review."""
    review_id: Optional[str]
    index_readiness_score: float = Field(..., ge=0, le=100)
    recommendation: str = Field(..., description="index, review, or reject")
    chunk_quality: str = Field(..., description="optimal, too_short, too_long, or empty")
    has_pii: bool
    pii_types: List[str]
    is_duplicate: bool = False  # Not checked in real-time API
    processing_time_ms: float
    metadata: Dict[str, Any]

class BatchReviewInput(BaseModel):
    """Input schema for batch review scoring."""
    reviews: List[ReviewInput] = Field(..., max_items=100, description="List of reviews (max 100)")

class BatchReviewOutput(BaseModel):
    """Output schema for batch scoring."""
    results: List[ReviewOutput]
    total_reviews: int
    avg_score: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float

class StatsResponse(BaseModel):
    """API statistics response."""
    total_requests: int
    total_reviews_scored: int
    avg_processing_time_ms: float
    started_at: str
    uptime_seconds: float

# ============================================================================
# SCORING LOGIC
# ============================================================================

def score_review(text: str, stars: Optional[float] = None) -> Dict[str, Any]:
    """
    Score a single review for index-readiness.
    
    Returns dict with:
    - chunk_quality_flag
    - chunk_score
    - has_pii
    - pii_types
    - pii_score
    - index_readiness_score
    - recommendation
    - metadata
    """
    start_time = time.time()
    
    # Chunk analysis
    chunks = chunk_text(text)
    chunk_analysis = analyze_chunk_quality(chunks)
    
    # Map quality_flag to chunk_quality_flag
    chunk_quality_flag = chunk_analysis.get('quality_flag', 'too_short')
    
    chunk_score = {
        'optimal': 100,
        'too_short': 50,
        'too_long': 0,
        'empty': 0
    }.get(chunk_quality_flag, 50)
    
    # PII detection
    has_pii, pii_types = detect_pii(text)
    pii_score = 0 if has_pii else 100
    
    # Duplicate score (always 100 for single review - no context to check)
    duplicate_score = 100
    
    # Overall score (weighted average)
    index_readiness_score = (
        chunk_score * 0.4 +
        duplicate_score * 0.3 +
        pii_score * 0.3
    )
    
    # Recommendation
    if index_readiness_score >= 70:
        recommendation = "index"
    elif index_readiness_score >= 50:
        recommendation = "review"
    else:
        recommendation = "reject"
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'chunk_quality_flag': chunk_quality_flag,
        'chunk_score': chunk_score,
        'chunk_count': chunk_analysis['chunk_count'],
        'avg_chunk_tokens': chunk_analysis['avg_chunk_tokens'],
        'has_pii': has_pii,
        'pii_types': pii_types,
        'pii_score': pii_score,
        'duplicate_score': duplicate_score,
        'index_readiness_score': round(index_readiness_score, 1),
        'recommendation': recommendation,
        'processing_time_ms': round(processing_time, 2),
        'metadata': {
            'stars': stars,
            'text_length': len(text),
            'chunk_count': chunk_analysis['chunk_count']
        }
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Index-Readiness API",
        "version": "1.0.0",
        "endpoints": {
            "POST /score": "Score a single review",
            "POST /batch": "Score multiple reviews",
            "GET /health": "Health check",
            "GET /stats": "API statistics",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.post("/score", response_model=ReviewOutput, tags=["Scoring"])
async def score_single_review(review: ReviewInput):
    """
    Score a single review for index-readiness.
    
    **Example:**
```json
    {
      "text": "Amazing food and great service!",
      "stars": 5,
      "review_id": "abc123"
    }
```
    
    **Returns:**
    - index_readiness_score (0-100)
    - recommendation (index/review/reject)
    - chunk_quality (optimal/too_short/too_long)
    - has_pii (boolean)
    - pii_types (list)
    - processing_time_ms
    """
    try:
        # Update global stats
        stats["total_requests"] += 1
        stats["total_reviews_scored"] += 1
        
        # Score the review
        result = score_review(review.text, review.stars)
        
        # Update avg processing time
        old_avg = stats["avg_processing_time_ms"]
        new_avg = (old_avg * (stats["total_reviews_scored"] - 1) + result['processing_time_ms']) / stats["total_reviews_scored"]
        stats["avg_processing_time_ms"] = round(new_avg, 2)
        
        # Build response
        return ReviewOutput(
            review_id=review.review_id,
            index_readiness_score=result['index_readiness_score'],
            recommendation=result['recommendation'],
            chunk_quality=result['chunk_quality_flag'],
            has_pii=result['has_pii'],
            pii_types=result['pii_types'],
            is_duplicate=False,
            processing_time_ms=result['processing_time_ms'],
            metadata=result['metadata']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/batch", response_model=BatchReviewOutput, tags=["Scoring"])
async def score_batch_reviews(batch: BatchReviewInput):
    """
    Score multiple reviews in one request (max 100).
    
    **Example:**
```json
    {
      "reviews": [
        {"text": "Great place!", "stars": 5},
        {"text": "Terrible service", "stars": 1}
      ]
    }
```
    
    **Returns:**
    - results: List of scored reviews
    - total_reviews: Count
    - avg_score: Average index-readiness score
    - processing_time_ms: Total batch time
    """
    try:
        start_time = time.time()
        
        # Update global stats
        stats["total_requests"] += 1
        stats["total_reviews_scored"] += len(batch.reviews)
        
        # Score all reviews
        results = []
        total_score = 0
        
        for review in batch.reviews:
            result = score_review(review.text, review.stars)
            
            results.append(ReviewOutput(
                review_id=review.review_id,
                index_readiness_score=result['index_readiness_score'],
                recommendation=result['recommendation'],
                chunk_quality=result['chunk_quality_flag'],
                has_pii=result['has_pii'],
                pii_types=result['pii_types'],
                is_duplicate=False,
                processing_time_ms=result['processing_time_ms'],
                metadata=result['metadata']
            ))
            
            total_score += result['index_readiness_score']
        
        batch_processing_time = (time.time() - start_time) * 1000
        avg_score = total_score / len(batch.reviews) if batch.reviews else 0
        
        return BatchReviewOutput(
            results=results,
            total_reviews=len(batch.reviews),
            avg_score=round(avg_score, 1),
            processing_time_ms=round(batch_processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and uptime.
    """
    start_time = datetime.fromisoformat(stats["started_at"])
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2)
    )

@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """
    Get API usage statistics.
    
    Returns:
    - total_requests
    - total_reviews_scored
    - avg_processing_time_ms
    - uptime_seconds
    """
    start_time = datetime.fromisoformat(stats["started_at"])
    uptime = (datetime.now() - start_time).total_seconds()
    
    return StatsResponse(
        total_requests=stats["total_requests"],
        total_reviews_scored=stats["total_reviews_scored"],
        avg_processing_time_ms=stats["avg_processing_time_ms"],
        started_at=stats["started_at"],
        uptime_seconds=round(uptime, 2)
    )

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log API startup."""
    print("="*70)
    print("RAG INDEX-READINESS API STARTED")
    print("="*70)
    print(f"Started at: {stats['started_at']}")
    print(f"Docs: http://localhost:8000/docs")
    print("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    """Log API shutdown."""
    print("="*70)
    print("RAG INDEX-READINESS API SHUTDOWN")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total reviews scored: {stats['total_reviews_scored']}")
    print("="*70)
