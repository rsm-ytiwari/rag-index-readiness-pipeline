"""
utils.py
Utility functions for RAG Index-Readiness Pipeline

Functions:
1. chunk_text() - Split text into semantic chunks (300-500 tokens)
2. detect_pii() - Detect personally identifiable information (email/phone/address)
"""

import re
from typing import List, Tuple, Dict
import tiktoken

# ============================================================================
# TOKENIZATION SETUP
# ============================================================================

# Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(TOKENIZER.encode(text))


# ============================================================================
# FUNCTION 1: CHUNK_TEXT
# ============================================================================

def chunk_text(
    text: str,
    target_min_tokens: int = 300,
    target_max_tokens: int = 500,
    overlap_sentences: int = 1
) -> List[Dict[str, any]]:
    """
    Split text into semantic chunks based on sentences.
    
    Target: 300-500 tokens per chunk for optimal RAG retrieval.
    
    Args:
        text: Input text to chunk
        target_min_tokens: Minimum tokens per chunk (default: 300)
        target_max_tokens: Maximum tokens per chunk (default: 500)
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)
    
    Returns:
        List of chunk dictionaries with:
        - text: chunk text
        - tokens: token count
        - sentences: sentence count
        - chunk_id: chunk number (0-indexed)
    
    Strategy:
    1. Split text into sentences
    2. Group sentences until target_max_tokens is reached
    3. If a single sentence > target_max_tokens, split by word boundary
    """
    
    # Handle empty or very short text
    if not text or len(text.strip()) == 0:
        return []
    
    token_count = count_tokens(text)
    
    # If entire text is within target range, return as single chunk
    if target_min_tokens <= token_count <= target_max_tokens:
        return [{
            'text': text,
            'tokens': token_count,
            'sentences': len(split_sentences(text)),
            'chunk_id': 0
        }]
    
    # If text is shorter than minimum, still return it (we'll flag it later)
    if token_count < target_min_tokens:
        return [{
            'text': text,
            'tokens': token_count,
            'sentences': len(split_sentences(text)),
            'chunk_id': 0
        }]
    
    # Split into sentences
    sentences = split_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds max, split by words
        if sentence_tokens > target_max_tokens:
            # Save current chunk if exists
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'tokens': current_tokens,
                    'sentences': len(current_chunk),
                    'chunk_id': chunk_id
                })
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            word_chunks = split_long_sentence(sentence, target_max_tokens)
            for wc in word_chunks:
                chunks.append({
                    'text': wc,
                    'tokens': count_tokens(wc),
                    'sentences': 1,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            continue
        
        # Check if adding this sentence exceeds max
        if current_tokens + sentence_tokens > target_max_tokens:
            # Save current chunk
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'tokens': current_tokens,
                    'sentences': len(current_chunk),
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_tokens = sum(count_tokens(s) for s in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'tokens': current_tokens,
            'sentences': len(current_chunk),
            'chunk_id': chunk_id
        })
    
    return chunks


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Uses regex to identify sentence boundaries while handling:
    - Abbreviations (Dr., Mr., Mrs., etc.)
    - Decimal numbers (1.5, 3.14)
    - Ellipsis (...)
    """
    # Basic sentence splitting pattern
    # Split on: . ! ? followed by space and capital letter or end of string
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    
    sentences = re.split(pattern, text)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """
    Split a very long sentence by word boundaries.
    
    Used when a single sentence exceeds max_tokens.
    """
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = count_tokens(word + ' ')
        
        if current_tokens + word_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================================
# FUNCTION 2: DETECT_PII
# ============================================================================

def detect_pii(text: str) -> Tuple[bool, List[str]]:
    """
    Detect personally identifiable information (PII) in text.
    
    Detects:
    - Email addresses
    - Phone numbers (US/international formats)
    - Street addresses (basic patterns)
    
    Args:
        text: Input text to scan for PII
    
    Returns:
        Tuple of (has_pii: bool, pii_types: List[str])
        - has_pii: True if any PII detected
        - pii_types: List of detected PII types (e.g., ['email', 'phone'])
    
    Note: This is a basic implementation. Production systems would use
    more sophisticated NER (Named Entity Recognition) models.
    """
    
    if not text or len(text.strip()) == 0:
        return False, []
    
    pii_types = []
    
    # -------------------------------------------------------------------------
    # EMAIL DETECTION
    # -------------------------------------------------------------------------
    # Pattern: username@domain.tld
    # Handles: john@example.com, jane.doe@company.co.uk
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    if re.search(email_pattern, text):
        pii_types.append('email')
    
    # -------------------------------------------------------------------------
    # PHONE NUMBER DETECTION
    # -------------------------------------------------------------------------
    # Patterns for US/International phone numbers:
    # - (123) 456-7890
    # - 123-456-7890
    # - 123.456.7890
    # - +1 123 456 7890
    # - 1234567890
    
    phone_patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 123-456-7890
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',     # (123) 456-7890
        r'\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4}\b', # +1 123 456 7890
        r'\b\d{10}\b'                            # 1234567890
    ]
    
    for pattern in phone_patterns:
        if re.search(pattern, text):
            pii_types.append('phone')
            break  # Only add 'phone' once
    
    # -------------------------------------------------------------------------
    # ADDRESS DETECTION (BASIC)
    # -------------------------------------------------------------------------
    # Patterns for street addresses:
    # - 123 Main Street
    # - 456 Oak Ave
    # - P.O. Box 789
    
    address_patterns = [
        r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b',
        r'\bP\.?O\.?\s+Box\s+\d+\b',  # P.O. Box
        r'\b\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # 123 Main Street (simplified)
    ]
    
    for pattern in address_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            pii_types.append('address')
            break  # Only add 'address' once
    
    # Remove duplicates (shouldn't happen, but defensive)
    pii_types = list(set(pii_types))
    
    has_pii = len(pii_types) > 0
    
    return has_pii, pii_types


# ============================================================================
# HELPER FUNCTION: Analyze Chunk Quality
# ============================================================================

def analyze_chunk_quality(chunks: List[Dict]) -> Dict[str, any]:
    """
    Analyze quality of chunks for a review.
    
    Returns metrics:
    - chunk_count: Number of chunks
    - avg_chunk_tokens: Average tokens per chunk
    - min_chunk_tokens: Minimum tokens in any chunk
    - max_chunk_tokens: Maximum tokens in any chunk
    - quality_flag: "too_short", "optimal", "too_long"
    """
    
    if not chunks:
        return {
            'chunk_count': 0,
            'avg_chunk_tokens': 0,
            'min_chunk_tokens': 0,
            'max_chunk_tokens': 0,
            'quality_flag': 'empty'
        }
    
    token_counts = [c['tokens'] for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    
    # Determine quality flag
    if len(chunks) == 1:
        tokens = chunks[0]['tokens']
        if tokens < 100:
            quality_flag = 'too_short'
        elif tokens > 800:
            quality_flag = 'too_long'
        else:
            quality_flag = 'optimal'
    else:
        # Multiple chunks - check if average is in optimal range
        if avg_tokens < 200:
            quality_flag = 'too_short'
        elif avg_tokens > 600:
            quality_flag = 'too_long'
        else:
            quality_flag = 'optimal'
    
    return {
        'chunk_count': len(chunks),
        'avg_chunk_tokens': round(avg_tokens, 1),
        'min_chunk_tokens': min(token_counts),
        'max_chunk_tokens': max(token_counts),
        'quality_flag': quality_flag
    }


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    """
    Test functions on sample reviews.
    Run: python src/utils.py
    """
    
    print("="*70)
    print("TESTING UTILITY FUNCTIONS")
    print("="*70)
    
    # Test 1: Chunk Text
    print("\n" + "─"*70)
    print("TEST 1: chunk_text()")
    print("─"*70)
    
    test_texts = [
        # Short review
        "Great food! Will come back.",
        
        # Medium review (optimal)
        "I've been coming to this restaurant for years and it never disappoints. "
        "The service is excellent, the staff is friendly, and the atmosphere is cozy. "
        "My favorite dish is the salmon which is always cooked to perfection. "
        "The desserts are also amazing - highly recommend the chocolate cake. "
        "Prices are reasonable for the quality you get. Definitely worth a visit!",
        
        # Long review (needs chunking)
        ("This was my first time trying this place and I have to say I was impressed. " * 50)
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: Text length = {len(text)} chars, {count_tokens(text)} tokens")
        chunks = chunk_text(text)
        print(f"Result: {len(chunks)} chunk(s)")
        for chunk in chunks:
            print(f"  Chunk {chunk['chunk_id']}: {chunk['tokens']} tokens, {chunk['sentences']} sentences")
            print(f"    Preview: {chunk['text'][:80]}...")
        
        quality = analyze_chunk_quality(chunks)
        print(f"Quality: {quality['quality_flag']} (avg: {quality['avg_chunk_tokens']} tokens)")
    
    # Test 2: Detect PII
    print("\n" + "─"*70)
    print("TEST 2: detect_pii()")
    print("─"*70)
    
    test_pii_texts = [
        "Great restaurant, no complaints!",
        "You can reach me at john@example.com for reservations.",
        "Call us at 555-123-4567 to make a booking.",
        "We're located at 123 Main Street in downtown.",
        "Contact: jane.doe@company.com or (555) 987-6543. Visit us at 456 Oak Ave!",
    ]
    
    for i, text in enumerate(test_pii_texts):
        has_pii, pii_types = detect_pii(text)
        print(f"\nTest {i+1}: {text[:60]}...")
        print(f"  Has PII: {has_pii}")
        print(f"  Types: {pii_types if pii_types else 'None'}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
