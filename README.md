# 🛡️ Spam / Fraud Message Detection System

An end-to-end NLP-based classification system for detecting spam and fraudulent messages in SMS, emails, and other text communications. Built for fintech and telecom sectors to protect users from phishing attacks and financial fraud.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project implements a multi-model approach combining traditional machine learning (TF-IDF + Naive Bayes/SVM) with deep learning (BERT) to achieve high-accuracy spam detection with real-time performance.

**Key Features:**
- 🎯 **Multi-Model Detection**: TF-IDF baseline (fast) + BERT (high accuracy)
- ⚡ **Real-time API**: FastAPI service with <300ms latency
- 📊 **Interactive Dashboard**: Streamlit-based visualization and monitoring
- 🔒 **Privacy-First**: No PII storage, encrypted communication
- 🐳 **Production-Ready**: Docker containerization, cloud deployment support

## 🎯 Success Metrics

- **Model F1-Score**: ≥ 92%
- **Precision**: > 95% (minimize false positives)
- **API Latency**: < 300ms per request
- **Throughput**: 10,000+ messages/day

## 🏗️ Project Structure

```
spam_fraud_message/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and vectorized data
│   └── models/           # Trained model artifacts
├── src/
│   ├── preprocessing/    # Text cleaning and normalization
│   ├── features/         # TF-IDF and BERT feature extraction
│   ├── models/           # Model training and inference
│   ├── api/              # FastAPI endpoints
│   └── utils/            # Configuration and helpers
├── scripts/              # Setup and training scripts
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebooks for exploration
├── dashboard/            # Streamlit dashboard
├── logs/                 # Training and API logs
├── reports/              # Evaluation reports and metrics
└── technical_guides/     # Detailed implementation guides

```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/selftaughtrahul/spam-detection.git
cd spam_fraud_message

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP resources
python scripts/setup_nlp.py
```

### 2. Prepare Data

Place your spam datasets in `data/raw/` (e.g., SMS Spam Collection, Enron Email dataset)

### 3. Train Models

```bash
# Train baseline TF-IDF + Naive Bayes
python scripts/train_baseline.py

# Fine-tune BERT model (requires GPU)
python scripts/train_bert.py
```

### 4. Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## 📡 API Usage

### Single Message Detection

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "message_text": "Congratulations! You won $1000. Click here: bit.ly/spam",
    "metadata": {
      "source": "SMS",
      "sender_id": "+123456789"
    }
  }'
```

**Response:**
```json
{
  "prediction": "Spam",
  "confidence": 0.98,
  "risk_level": "High",
  "features_detected": ["link_present", "urgency_detected"],
  "model": "BERT-v1"
}
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/v1/batch-detect" \
  -F "file=@messages.csv"
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📊 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.9+ |
| **NLP** | NLTK, spaCy, scikit-learn |
| **Deep Learning** | PyTorch, Transformers (Hugging Face) |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Deployment** | Docker, AWS/GCP |

## 📚 Documentation

Comprehensive technical guides are available in [`technical_guides/`](technical_guides/):

1. [Project Setup](technical_guides/01_PROJECT_SETUP.md)
2. [Preprocessing Module](technical_guides/02_PREPROCESSING_MODULE.md)
3. [Feature Extraction](technical_guides/03_FEATURE_EXTRACTION.md)
4. [Model Training](technical_guides/04_MODEL_TRAINING.md)
5. [API Development](technical_guides/05_API_DEVELOPMENT.md)
6. [Dashboard](technical_guides/06_DASHBOARD.md)
7. [Deployment](technical_guides/07_DEPLOYMENT.md)
8. [Testing](technical_guides/08_TESTING.md)
9. [Troubleshooting](technical_guides/09_TROUBLESHOOTING.md)

## 🐳 Docker Deployment

```bash
# Build image
docker build -t spam-detector .

# Run container
docker run -p 8000:8000 spam-detector
```

## 🔒 Security & Privacy

- No storage of message content beyond processing
- HTTPS-only API communication
- PII scrubbing before logging
- Configurable data retention policies

## 📈 Performance

| Model | Accuracy | F1-Score | Latency |
|-------|----------|----------|---------|
| Naive Bayes + TF-IDF | 85% | 83% | <50ms |
| BERT (fine-tuned) | 95% | 94% | <300ms |

## 🛣️ Roadmap

- [ ] Multi-language support
- [ ] Image-based spam detection (OCR)
- [ ] Real-time SMS gateway integration
- [ ] Model drift monitoring
- [ ] A/B testing framework

## 📄 Project Documents

- [Statement of Work (SOW)](01_SOW.md)
- [Business Requirements (BRD)](02_BRD.md)
- [Functional Requirements (FRD)](03_FRD.md)

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Rahul** - [GitHub](https://github.com/selftaughtrahul)

## 🙏 Acknowledgments

- SMS Spam Collection Dataset
- Hugging Face Transformers
- FastAPI and Streamlit communities

---

**Status**: 🚧 In Development | **Version**: 0.1.0 | **Last Updated**: February 2026
