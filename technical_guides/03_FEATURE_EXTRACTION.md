# Technical Guide 03: Feature Extraction
## Spam / Fraud Message Detection System

---

## 📋 Overview
To classify messages, we must convert text into numerical format. We use two levels of features: Fast (TF-IDF) and Contextual (BERT).

## Step 1: TF-IDF Vectorization
Create `src/features/tfidf_builder.py`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def train_tfidf(corpus, save_path):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(corpus)
    
    # Save for inference
    with open(save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return X, vectorizer
```

## Step 2: BERT Embeddings
Create `src/features/bert_embeddings.py`:

```python
from transformers import BertTokenizer, BertModel
import torch

class BERTExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]
```

## ✅ Checklist
- [ ] TF-IDF Pickle saved
- [ ] BERT Tokenizer local cache setup
- [ ] Embedding dimensions verified (768 for BERT)
