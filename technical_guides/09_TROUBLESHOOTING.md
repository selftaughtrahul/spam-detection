# Technical Guide 09: Troubleshooting
## Spam / Fraud Message Detection System

---

## 📋 Common Issues

### 1. CUDA/GPU Errors
- **Symptom:** BERT training crashes with `RuntimeError: CUDA out of memory`.
- **Fix:** Reduce `per_device_train_batch_size` to 4 or 2 in `train_bert.py`.

### 2. Slow Inference
- **Symptom:** API takes > 2 seconds per request.
- **Fix:** Use `DistilBERT` instead of full BERT, or use ONNX Runtime.

### 3. False Positives
- **Symptom:** Critical transaction codes marked as spam.
- **Fix:** Add a "Whitelist" layer in `src/utils/rules.py` for known secure patterns.

---

## ✅ Contact Support
For further issues, check the `logs/error.log` file.
