"""
Configuration settings for the Spam/Fraud Detection System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Spam/Fraud Detection Configuration
SPAM_THRESHOLD = 0.8  # Probabilistic threshold for blocking
FRAUD_THRESHOLD = 0.9  # Higher threshold for fraud classification

# Model settings

# Model settings
MODEL_CONFIGS = {
    "naive_bayes": {
        "alpha": 0.1
    },
    "svm": {
        "C": 2.0,
        "max_iter": 1000,
        "solver": "lbfgs"
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1
    },
    "bert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
        "epochs": 4,
        "learning_rate": 2e-5,
        "warmup_steps": 20
    }
}

# TF-IDF settings
TFIDF_CONFIG = {
    "max_features": 20000,
    "ngram_range": (1, 3),
    "min_df": 5,
    "max_df": 0.8
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

