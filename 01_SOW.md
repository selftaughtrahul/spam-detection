# Statement of Work (SOW)
## Spam / Fraud Message Detection System

---

### 1. Project Overview

**Project Name:** Spam / Fraud Message Detection System

**Project Duration:** 8-10 weeks

**Project Type:** NLP-based Classification System

**Objective:** Develop an end-to-end spam and fraud message detection system capable of identifying malicious SMS, emails, and phishing messages to protect users in fintech and telecom sectors.

---

### 2. Project Background

In the digital era, spam and fraudulent messages have become a significant threat to security and privacy. Telecom and Fintech companies face increasing challenges in protecting their customers from phishing attacks, financial fraud, and unsolicited promotional messages. Manual filtering is no longer viable due to the sheer volume of messaging data. This project aims to automate detection to help businesses:

- Protect users from financial fraud and phishing
- Improve user experience by filtering out spam
- Enhance brand trust and security reputation
- Comply with telecommunication regulations regarding unsolicited messages

---

### 3. Project Scope

#### 3.1 In-Scope

- **Data Collection & Preprocessing**
  - Collect datasets (SMS Spam Collection, Enron Email, Phishing datasets)
  - Text cleaning (removing links, special characters, normalization)
  - Handling imbalanced data (Spam is often a minority class)

- **Model Development**
  - Implement Classification models (Naive Bayes, SVM)
  - Feature extraction using TF-IDF vectorization
  - Advanced deep learning models using BERT for contextual understanding
  - Model evaluation using Precision-Recall and F1-Score

- **API Development**
  - RESTful API using FastAPI
  - Real-time classification endpoint
  - Probabilistic scoring for fraud risk assessment

- **User Interface**
  - Web dashboard for monitoring spam trends
  - Interactive prediction interface for single messages
  - Batch processing interface for bulk analysis

- **Deployment**
  - Containerization using Docker
  - Cloud deployment setup
  - API documentation (Swagger/ReDoc)

#### 3.2 Out-of-Scope

- Voice-based fraud detection (vishing)
- Real-time network-level SMS gateway integration
- Multi-language support (Phase 2)
- Image-based spam detection (OCR)

---

### 4. Deliverables

| # | Deliverable | Description | Timeline |
|---|-------------|-------------|----------|
| 1 | Preprocessing Pipeline | Scalable pipeline for cleaning messaging data | Week 1-2 |
| 2 | TF-IDF Baseline Models | Naive Bayes/SVM models with vectorization | Week 3-4 |
| 3 | BERT Detection Model | Fine-tuned BERT model for high-accuracy detection | Week 5-6 |
| 4 | Spam Detection API | FastAPI based service for real-time detection | Week 6-7 |
| 5 | Monitoring Dashboard | Streamlit UI for data visualization | Week 7-8 |
| 6 | Technical Documentation | Full implementation guides and API docs | Week 8-9 |
| 7 | Deployment Package | Dockerized solution for cloud deployment | Week 9-10 |

---

### 5. Technical Stack

**Programming Language:** Python 3.9+

**NLP Libraries:**
- NLTK, spaCy
- Scikit-learn (TF-IDF, ML Models)
- Transformers (BERT - Hugging Face)

**Deep Learning:**
- PyTorch / TensorFlow

**Web Framework:**
- FastAPI
- Streamlit (Dashboard)

**Infrastructure:**
- Docker
- AWS / GCP

---

### 6. Project Milestones

| Milestone | Description | Target Date |
|-----------|-------------|-------------|
| M1 | Environment Setup & Data Loading | End of Week 1 |
| M2 | Data Cleaning & EDA Complete | End of Week 2 |
| M3 | TF-IDF Models Baseline achieved | End of Week 4 |
| M4 | BERT Model Training Complete | End of Week 6 |
| M5 | API Layer Development Complete | End of Week 7 |
| M6 | Dashboard Visualization Complete | End of Week 8 |
| M7 | Testing & Refinement | End of Week 9 |
| M8 | Final Documentation & Handoff | End of Week 10 |

---

### 7. Success Criteria

- Model F1-Score ≥ 92% (High precision is critical to avoid False Positives)
- API Latency < 300ms for single message check
- Zero downtime during standard batch processing
- 100% documentation coverage for core modules

---

### 8. Assumptions & Constraints

**Assumptions:**
- Availability of labeled spam/ham datasets.
- Access to computational resources (GPU) for BERT training.

**Constraints:**
- Data privacy: No Personal Identifiable Information (PII) should be stored.
- Model size: BERT base models might require significant memory.

---

### 9. Approval

**Prepared by:** Antigravity AI  
**Date:** February 13, 2026  
**Status:** Approved (Implementation Plan)
