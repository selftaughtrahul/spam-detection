# Technical Guide 05: API Development
## Spam / Fraud Message Detection System

---

## 📋 Overview
Expose the models as a REST API using FastAPI for real-time integration.

## Step 1: API Implementation
Create `src/api/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.preprocessing import clean_message
from src.models.predictor import SpamPredictor

app = FastAPI(title="Spam & Fraud Detection API")
predictor = SpamPredictor()

class Message(BaseModel):
    text: str

@app.post("/predict")
async def predict_spam(message: Message):
    cleaned = clean_message(message.text)
    prediction, confidence = predictor.predict(cleaned)
    
    return {
        "is_spam": bool(prediction),
        "confidence": float(confidence),
        "cleaned_text": cleaned
    }

@app.get("/health")
def health():
    return {"status": "active"}
```

## Step 2: Run Server
```bash
uvicorn src.api.main:app --reload
```

## ✅ Checklist
- [ ] Swagger documentation accessible at `/docs`
- [ ] Health check returns 200 OK
- [ ] Prediction response latency < 500ms
