# Technical Guide 02: Preprocessing Module
## Spam / Fraud Message Detection System

---

## 📋 Overview
Messaging data is often messy, containing URLs, phone numbers, and specialized slang. This module cleans and prepares raw text for classification.

## Step 1: Core Cleaner
Create `src/preprocessing/cleaner.py`:

```python
import re
import string
from nltk.corpus import stopwords

def clean_message(text: str) -> str:
    """
    Cleans message text for spam detection.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Extract and sanitize URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
    
    # 3. Extract and sanitize Phone Numbers
    text = re.sub(r'\+?\d{10,}', ' [PHONE] ', text)
    
    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove extra whitespace
    text = " ".join(text.split())
    
    return text
```

## Step 2: Spam Pattern Identification
Spam often uses specific patterns that we can flag as features:

```python
def extract_spam_patterns(text: str) -> dict:
    """Detects spam-specific visual patterns."""
    return {
        "has_urgency": bool(re.search(r'urgent|act now|immediate', text, re.I)),
        "has_currency": bool(re.search(r'\$|£|€', text)),
        "excessive_caps": sum(1 for c in text if c.isupper()) / len(text) > 0.3 if text else False
    }
```

## Step 3: Integration
Create `src/preprocessing/__init__.py`:
```python
from .cleaner import clean_message, extract_spam_patterns
```

## ✅ Checklist
- [ ] URL sanitization tested
- [ ] Phone number masking verified
- [ ] Stopword removal optionality check
