import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# Spam Configuration
SPAM_THRESHOLD = 0.8  # Probabilistic threshold for blocking
BERT_MODEL_NAME = "bert-base-uncased"