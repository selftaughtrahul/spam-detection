"""
Complete Dynamic Training Script
Trains all 4 models (Naive Bayes, SVM, XGBoost, DistilBERT) with comprehensive evaluation
Based on notebook implementation with production-ready code structure
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from scipy.special import softmax
from src.utils.config import RAW_DATA_DIR, MODELS_DIR, BASE_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpamModelTrainer:
    """Complete training pipeline for all spam detection models"""
    
    def __init__(self, data_path=None):
        """Initialize trainer"""
        self.data_path = data_path or RAW_DATA_DIR / "spam_data.csv"
        self.models_dir = MODELS_DIR
        self.reports_dir = BASE_DIR / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.vectorizer = None
        self.model_nb = None
        self.model_xgb = None
        self.model_svm = None
        self.model_svm_cal = None
        self.model_bert = None
        self.tokenizer_bert = None
        
        # Results storage
        self.results = {}
        
        logger.info("Initialized SpamModelTrainer")
    
    def load_data(self):
        """Load and split data"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        logger.info(f"Total Dataset: {len(df)}")
        logger.info(f"Total Spam: {(df['label_binary']==1).sum()}")
        logger.info(f"Total Ham: {(df['label_binary']==0).sum()}")
        
        X = df['message']
        y = df['label_binary']
        
        # Split: 70% train, 15% val, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_naive_bayes(self, X_train, X_test, y_train, y_test):
        """Train Naive Bayes model"""
        logger.info("="*60)
        logger.info("TRAINING NAIVE BAYES")
        logger.info("="*60)
        
        # Vectorization
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        self.model_nb = MultinomialNB()
        self.model_nb.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = self.model_nb.predict(X_test_tfidf)
        y_pred_proba = self.model_nb.predict_proba(X_test_tfidf)[:, 1]
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba, "Naive Bayes")
        self.results['Naive Bayes'] = results
        
        # Save model
        with open(self.models_dir / "naive_bayes.pkl", 'wb') as f:
            pickle.dump(self.model_nb, f)
        with open(self.models_dir / "tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info("Naive Bayes model saved")
        return results, X_train_tfidf, X_test_tfidf
    
    def train_xgboost(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        """Train XGBoost model"""
        logger.info("="*60)
        logger.info("TRAINING XGBOOST")
        logger.info("="*60)
        
        # Calculate class weight
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train model
        self.model_xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )
        self.model_xgb.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = self.model_xgb.predict(X_test_tfidf)
        y_pred_proba = self.model_xgb.predict_proba(X_test_tfidf)[:, 1]
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba, "XGBoost")
        self.results['XGBoost'] = results
        
        # Save model
        with open(self.models_dir / "xgboost.pkl", 'wb') as f:
            pickle.dump(self.model_xgb, f)
        
        logger.info("XGBoost model saved")
        return results
    
    def train_svm(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        """Train LinearSVC model"""
        logger.info("="*60)
        logger.info("TRAINING LINEAR SVC")
        logger.info("="*60)
        
        # Train model
        self.model_svm = LinearSVC(class_weight="balanced", random_state=42)
        self.model_svm.fit(X_train_tfidf, y_train)
        
        # Calibrate for probabilities
        self.model_svm_cal = CalibratedClassifierCV(self.model_svm, method="sigmoid", cv=5)
        self.model_svm_cal.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = self.model_svm.predict(X_test_tfidf)
        y_pred_proba = self.model_svm_cal.predict_proba(X_test_tfidf)[:, 1]
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba, "LinearSVC")
        self.results['LinearSVC'] = results
        
        # Save model
        with open(self.models_dir / "svm.pkl", 'wb') as f:
            pickle.dump(self.model_svm, f)
        with open(self.models_dir / "svm_calibrated.pkl", 'wb') as f:
            pickle.dump(self.model_svm_cal, f)
        
        logger.info("LinearSVC model saved")
        return results
    
    def train_distilbert(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train DistilBERT model"""
        logger.info("="*60)
        logger.info("TRAINING DISTILBERT")
        logger.info("="*60)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Model and tokenizer
        MODEL_NAME = "distilbert-base-uncased"
        self.tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        
        # Tokenize
        def tokenize(texts):
            return self.tokenizer_bert(
                list(texts),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
        
        train_encodings = tokenize(X_train)
        val_encodings = tokenize(X_val)
        test_encodings = tokenize(X_test)
        
        # Dataset class
        class SpamDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = np.array(labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = SpamDataset(train_encodings, y_train)
        val_dataset = SpamDataset(val_encodings, y_val)
        test_dataset = SpamDataset(test_encodings, y_test)
        
        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        ).to(device)
        
        # Class weights
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(logits.device)
                loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
                return (loss, outputs) if return_outputs else loss
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.models_dir / "distilbert_results"),
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            logging_steps=50,
            logging_dir=str(BASE_DIR / "logs"),
            report_to="none",
        )
        
        # Metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            proba = softmax(logits, axis=1)[:, 1]
            
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, proba)
            
            return {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc
            }
        
        # Train
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        
        logger.info("Test Set Results:")
        for metric, value in test_results.items():
            logger.info(f"  {metric:<30} {value:.4f}")
        
        # Save model
        model.save_pretrained(self.models_dir / "distilbert_model")
        self.tokenizer_bert.save_pretrained(self.models_dir / "distilbert_model")
        
        # Store results
        self.results['DistilBERT'] = {
            'accuracy': test_results['eval_accuracy'],
            'precision': test_results['eval_precision'],
            'recall': test_results['eval_recall'],
            'f1': test_results['eval_f1'],
            'roc_auc': test_results['eval_roc_auc']
        }
        
        logger.info("DistilBERT model saved")
        return self.results['DistilBERT']
    
    def _calculate_metrics(self, y_test, y_pred, y_pred_proba, model_name):
        """Calculate all metrics"""
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def create_comparison_report(self):
        """Create comprehensive comparison report"""
        logger.info("Creating comparison report...")
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            model: [
                results['accuracy'],
                results['precision'],
                results['recall'],
                results['f1'],
                results['roc_auc']
            ]
            for model, results in self.results.items()
        }, index=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
        
        # Save to CSV
        comparison.to_csv(self.reports_dir / "model_comparison.csv")
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison.to_string())
        print("="*80)
        
        # Find best model
        best_model = comparison.loc['F1 Score'].idxmax()
        best_f1 = comparison.loc['F1 Score', best_model]
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")
        
        return comparison
    
    def create_visualizations(self, y_test):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['Blues', 'Oranges', 'Greens', 'Purples']
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            # Confusion Matrix
            sns.heatmap(
                results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap=colors[idx % len(colors)],
                ax=axes[0, idx],
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"]
            )
            axes[0, idx].set_xlabel("Predicted Label")
            axes[0, idx].set_ylabel("Actual Label")
            axes[0, idx].set_title(f"{model_name}\nConfusion Matrix")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            axes[1, idx].plot(fpr, tpr, lw=2, label=f"AUC = {results['roc_auc']:.4f}")
            axes[1, idx].plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
            axes[1, idx].set_xlabel("False Positive Rate")
            axes[1, idx].set_ylabel("True Positive Rate")
            axes[1, idx].set_title(f"{model_name}\nROC Curve")
            axes[1, idx].legend()
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / "all_models_evaluation.png", dpi=150)
        logger.info(f"Saved visualization to {self.reports_dir / 'all_models_evaluation.png'}")
        plt.close()
    
    def run_all(self, include_bert=True):
        """Run complete training pipeline"""
        logger.info("="*80)
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train traditional ML models
        nb_results, X_train_tfidf, X_test_tfidf = self.train_naive_bayes(
            X_train, X_test, y_train, y_test
        )
        
        xgb_results = self.train_xgboost(
            X_train_tfidf, X_test_tfidf, y_train, y_test
        )
        
        svm_results = self.train_svm(
            X_train_tfidf, X_test_tfidf, y_train, y_test
        )
        
        # Train BERT if requested
        if include_bert:
            bert_results = self.train_distilbert(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
        
        # Create reports
        comparison = self.create_comparison_report()
        self.create_visualizations(y_test)
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        
        return comparison


if __name__ == "__main__":
    import sys
    
    # Ask user about BERT
    print("\n" + "="*80)
    print("SPAM DETECTION - COMPLETE TRAINING PIPELINE")
    print("="*80)
    print("\nThis will train all 4 models:")
    print("  1. Naive Bayes (fast)")
    print("  2. XGBoost (gradient boosting)")
    print("  3. LinearSVC (SVM)")
    print("  4. DistilBERT (deep learning - slower)")
    print("\n" + "="*80)
    
    include_bert = input("\nInclude DistilBERT training? (takes longer) [y/N]: ").strip().lower()
    
    # Run training
    trainer = SpamModelTrainer()
    comparison = trainer.run_all(include_bert=(include_bert == 'y'))
    
    print("\n✅ All models trained and saved!")
    print(f"📊 Results saved to: {trainer.reports_dir}")
    print(f"💾 Models saved to: {trainer.models_dir}")
