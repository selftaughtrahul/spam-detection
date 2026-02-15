# Technical Guide 07: Deployment
## Spam / Fraud Message Detection System

---

## 📋 Overview
Deployment guide using Docker and CI/CD concepts.

## Step 1: Dockerization
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]
```

## Step 2: Build & Run
```bash
docker build -t spam-detector:v1 .
docker run -p 80:80 spam-detector:v1
```

## ✅ Checklist
- [ ] Docker image built successfully
- [ ] Container ports mapped
- [ ] BERT model included in COPY (or mounted via volume)
