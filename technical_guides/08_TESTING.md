# Technical Guide 08: Testing
## Spam / Fraud Message Detection System

---

## 📋 Overview
Ensuring robustness through unit tests and performance benchmarks.

## Step 1: Unit Testing
Create `tests/test_preprocessing.py`:
```python
from src.preprocessing import clean_message

def test_url_removal():
    assert "[URL]" in clean_message("Click here: http://bit.ly")

def test_phone_masking():
    assert "[PHONE]" in clean_message("Call me at 9876543210")
```

## Step 2: Performance Testing
Benchmark BERT latency:
```bash
pytest tests/benchmark_latency.py
```

## ✅ Checklist
- [ ] All preprocessing steps tested
- [ ] Edge cases (empty string, emojis) handled
- [ ] Model inference speed verified
