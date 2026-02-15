# 🛡️ Spam Shield Pro

A production-ready, NLP-based Spam & Fraud Detection System.

## 🎯 Project Overview
This system detects spam and fraud in SMS/Email messages using a multi-model approach (Naive Bayes, XGBoost, LinearSVC, DistilBERT), achieving >99% accuracy. It features a standalone interactive dashboard for real-time analysis.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional if already trained)
```bash
python train_all_models.py
```
This script trains all models and saves them to `data/models/`.

### 3. Run Dashboard
```bash
streamlit run dashboard/app.py
```
This launches the web interface at `http://localhost:8501`.

## 🏗️ Technical Architecture

### Core Components
- **Training Engine** (`train_all_models.py`): Orchestrates data loading, preprocessing, model training, and evaluation.
- **Prediction Engine** (`src/models/predictor.py`): Dynamically loads trained models for inference.
- **Dashboard** (`dashboard/app.py`): Streamlit-based UI for real-time interaction.

### Models Implemented
| Model | Type | Feature Extraction | Best For |
|-------|------|--------------------|----------|
| **Naive Bayes** | Probabilistic | TF-IDF | Speed |
| **XGBoost** | Gradient Boosting | TF-IDF | Robustness |
| **LinearSVC** | SVM | TF-IDF | Balanced Performance |
| **DistilBERT** | Transformer (DL) | Token Embeddings | Max Accuracy (>99%) |

## 📁 Directory Structure
```
spam_fraud_message/
├── dashboard/          # 📊 Streamlit App
├── src/                # 🧠 Source Code (Models, Features, Utils)
├── data/               # 💾 Raw Data & Saved Models
├── reports/            # 📈 Performance Metrics & Plots
├── notebook/           # 📓 Experiments
└── train_all_models.py # 🎯 Main Training Script
```
