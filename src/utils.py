"""
Utility functions for the RAG pipeline.
Includes functions for text chunking, PII detection, and data validation.
"""

import re
from typing import List, Dict, Any
import pandas as pd


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better RAG retrieval.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def detect_pii(text: str) -> Dict[str, bool]:
    """
    Detect potential PII (Personally Identifiable Information) in text.

    Args:
        text: Input text to check

    Returns:
        Dictionary with PII detection results
    """
    if not text:
        return {
            "has_email": False,
            "has_phone": False,
            "has_ssn": False,
            "has_credit_card": False,
        }

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    has_email = bool(re.search(email_pattern, text))

    # Phone pattern (US format)
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    has_phone = bool(re.search(phone_pattern, text))

    # SSN pattern
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    has_ssn = bool(re.search(ssn_pattern, text))

    # Credit card pattern (simple check for 16 digits)
    cc_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    has_credit_card = bool(re.search(cc_pattern, text))

    return {
        "has_email": has_email,
        "has_phone": has_phone,
        "has_ssn": has_ssn,
        "has_credit_card": has_credit_card,
    }


def remove_pii(text: str) -> str:
    """
    Remove or mask PII from text.

    Args:
        text: Input text

    Returns:
        Text with PII masked
    """
    if not text:
        return text

    # Mask email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
    )

    # Mask phone
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

    # Mask SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)

    # Mask credit card
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CREDIT_CARD]", text)

    return text


def validate_review_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate review data quality.

    Args:
        df: DataFrame with review data

    Returns:
        Dictionary with validation results
    """
    validation = {
        "total_records": len(df),
        "missing_text": df["text"].isna().sum() if "text" in df.columns else 0,
        "missing_stars": df["stars"].isna().sum() if "stars" in df.columns else 0,
        "missing_business_id": df["business_id"].isna().sum()
        if "business_id" in df.columns
        else 0,
        "empty_text": (df["text"].str.strip() == "").sum()
        if "text" in df.columns
        else 0,
    }

    if "text" in df.columns:
        validation["avg_text_length"] = df["text"].str.len().mean()
        validation["min_text_length"] = df["text"].str.len().min()
        validation["max_text_length"] = df["text"].str.len().max()

    if "stars" in df.columns:
        validation["avg_stars"] = df["stars"].mean()
        validation["star_distribution"] = df["stars"].value_counts().to_dict()

    return validation


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

    return text.strip()


def compute_text_stats(text: str) -> Dict[str, Any]:
    """
    Compute statistics for text analysis.

    Args:
        text: Input text

    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {"length": 0, "word_count": 0, "sentence_count": 0, "avg_word_length": 0}

    words = text.split()
    sentences = re.split(r"[.!?]+", text)

    return {
        "length": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
    }
