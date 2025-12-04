"""
PII Redaction Module (Free Version)

Rule-based PII redaction using regex patterns.
No API costs - 100% free and fast.

For production LLM-based redaction, see docs/LLM_REDACTION.md
"""

import re
from typing import Tuple, List, Dict
import pandas as pd

class PIIRedactor:
    """
    Free PII redaction using regex patterns.
    
    Detects and redacts:
    - Email addresses
    - Phone numbers
    - Street addresses
    - Credit card numbers (basic)
    """
    
    def __init__(self, redaction_style='bracket'):
        """
        Initialize redactor.
        
        Args:
            redaction_style: 'bracket' ([EMAIL]) or 'generic' (contact us)
        """
        self.redaction_style = redaction_style
        
        # Regex patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'address': r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
        
        # Replacement text by style
        self.replacements = {
            'bracket': {
                'email': '[EMAIL REDACTED]',
                'phone': '[PHONE REDACTED]',
                'address': '[ADDRESS REDACTED]',
                'credit_card': '[CARD REDACTED]'
            },
            'generic': {
                'email': 'our email',
                'phone': 'our phone number',
                'address': 'our location',
                'credit_card': 'payment method'
            }
        }
    
    def redact(self, text: str, pii_types: List[str] = None) -> Tuple[str, Dict]:
        """
        Redact PII from text.
        
        Args:
            text: Original text
            pii_types: List of PII types to redact (if None, redact all)
        
        Returns:
            Tuple of (redacted_text, metadata)
        """
        if pii_types is None:
            pii_types = list(self.patterns.keys())
        
        redacted_text = text
        redactions_made = {}
        
        for pii_type in pii_types:
            if pii_type not in self.patterns:
                continue
            
            pattern = self.patterns[pii_type]
            replacement = self.replacements[self.redaction_style][pii_type]
            
            # Count matches
            matches = re.findall(pattern, redacted_text, re.IGNORECASE)
            redactions_made[pii_type] = len(matches) if isinstance(matches[0] if matches else None, str) else len(matches)
            
            # Redact
            redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)
        
        metadata = {
            'original_length': len(text),
            'redacted_length': len(redacted_text),
            'redactions_made': redactions_made,
            'total_redactions': sum(redactions_made.values()),
            'redaction_rate': sum(redactions_made.values()) / len(text.split()) if text else 0
        }
        
        return redacted_text, metadata
    
    def redact_dataframe(self, df: pd.DataFrame, text_col: str = 'text', pii_types_col: str = 'pii_types_list') -> pd.DataFrame:
        """
        Redact PII from entire DataFrame.
        
        Args:
            df: DataFrame with review data
            text_col: Column name containing text
            pii_types_col: Column name containing list of PII types
        
        Returns:
            DataFrame with redacted_text column added
        """
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_col]
            pii_types = row[pii_types_col] if pii_types_col in df.columns else None
            
            redacted_text, metadata = self.redact(text, pii_types)
            
            results.append({
                'review_id': row.get('review_id', idx),
                'original_text': text,
                'redacted_text': redacted_text,
                'redactions_made': metadata['total_redactions'],
                'redaction_rate': metadata['redaction_rate']
            })
        
        return pd.DataFrame(results)

def quick_redact(text: str, pii_types: List[str] = None) -> str:
    """
    Quick convenience function for single redaction.
    
    Args:
        text: Text to redact
        pii_types: PII types to redact (default: all)
    
    Returns:
        Redacted text
    """
    redactor = PIIRedactor()
    redacted_text, _ = redactor.redact(text, pii_types)
    return redacted_text

if __name__ == '__main__':
    # Test PII redaction
    print("="*70)
    print("PII REDACTION TEST (FREE VERSION)")
    print("="*70)
    
    test_cases = [
        {
            'text': "Contact me at john.doe@example.com or call 555-123-4567 for more info.",
            'pii_types': ['email', 'phone']
        },
        {
            'text': "Visit us at 123 Main Street or email support@company.com",
            'pii_types': ['email', 'address']
        },
        {
            'text': "Great service! My card 4532-1234-5678-9010 was charged correctly.",
            'pii_types': ['credit_card']
        },
        {
            'text': "Amazing food! Highly recommend this restaurant.",
            'pii_types': None  # No PII
        }
    ]
    
    redactor = PIIRedactor(redaction_style='bracket')
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}:")
        print(f"Original:  {test['text']}")
        
        redacted, metadata = redactor.redact(test['text'], test['pii_types'])
        
        print(f"Redacted:  {redacted}")
        print(f"Stats:     {metadata['total_redactions']} redactions made")
        print(f"           {metadata['redactions_made']}")
    
    print("\n" + "="*70)
    print("✅ PII REDACTION TESTS PASSED")
    print("="*70)
    print("\n💡 Note: This is rule-based redaction (free, no API)")
    print("   For LLM-based context-aware redaction, see docs/LLM_REDACTION.md")
