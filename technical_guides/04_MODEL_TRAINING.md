# Technical Guide 04: Model Training
## Spam / Fraud Message Detection System

---

## 📋 Overview
This guide covers training a baseline Naive Bayes model and fine-tuning a BERT classifier for high-accuracy fraud detection.

## Step 1: Classical Machine Learning (Naive Bayes)
Create `src/models/naive_bayes.py`:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_nb(X_train, y_train, X_test, y_test):
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model
```

## Step 2: BERT Fine-Tuning
Create `scripts/train_bert.py`:

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

def fine_tune_bert(train_dataset, val_dataset):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    model.save_pretrained("data/models/spam_bert_v1")
```

## ✅ Checklist
- [ ] Naive Bayes baseline > 85% F1
- [ ] Training logs stored in `logs/`
- [ ] Model weights saved in `data/models/`
