# Technical Guide 01: Project Setup
## Spam / Fraud Message Detection System

---

## 📋 Overview
This guide covers the initial configuration, directory structure, and environment setup for the Spam/Fraud Detection project.

## Step 1: Create Project Structure
From `d:\Codebasics\NLP_Projects\spam_fraud_message\`, run:

```bash
# Data directories
mkdir data\raw data\processed data\models

# Source code
mkdir src\preprocessing src\features src\models src\api src\utils

# Support folders
mkdir scripts tests notebooks dashboard logs reports

# Initialize Python packages
type nul > src\__init__.py
type nul > src\preprocessing\__init__.py
type nul > src\features\__init__.py
type nul > src\models\__init__.py
type nul > src\api\__init__.py
type nul > src\utils\__init__.py
```

## Step 2: Environment Setup
Create `requirements.txt`:
```txt
# Data & NLP
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
spacy>=3.6.0
transformers>=4.30.0
torch>=2.0.0

# API & UI
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.25.0
python-multipart>=0.0.6

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
pytest>=7.4.0
```

Install:
```bash
pip install -r requirements.txt
```

## Step 3: NLP Resources
Create `scripts/setup_nlp.py`:
```python
import nltk
import spacy

def setup():
    nltk.download(['punkt', 'stopwords', 'wordnet'])
    try:
        spacy.cli.download("en_core_web_sm")
    except:
        pass

if __name__ == "__main__":
    setup()
```

## Step 4: Configuration
Create `src/utils/config.py`:
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# Spam Configuration
SPAM_THRESHOLD = 0.8  # Probabilistic threshold for blocking
BERT_MODEL_NAME = "bert-base-uncased"
```

## ✅ Checklist
- [ ] Folders created
- [ ] Dependencies installed
- [ ] NLP data downloaded
- [ ] Git initialized
