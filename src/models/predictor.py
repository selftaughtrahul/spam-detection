"""
Dynamic Model Predictor
Loads and uses any trained model for spam prediction
"""
import pickle
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.config import MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpamPredictor:
    """Dynamic predictor that can load and use any trained model"""
    
    AVAILABLE_MODELS = {
        'naive_bayes': 'naive_bayes.pkl',
        'xgboost': 'xgboost.pkl',
        'svm': 'svm_calibrated.pkl',
        'distilbert': 'final_spam_model'  # Updated to match user's folder name
    }
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize predictor
        
        Args:
            model_type: Type of model to load ('naive_bayes', 'xgboost', 'svm', 'distilbert')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self._load_model()
        logger.info(f"Loaded {model_type} model")
    
    def _load_model(self):
        """Load the specified model"""
        model_file = self.AVAILABLE_MODELS[self.model_type]
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Please train the model first using train_all_models.py"
            )
        
        if self.model_type == 'distilbert':
            # Load BERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            # Load traditional ML model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer (required for all traditional models)
            vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
            if not vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer not found: {vectorizer_path}. "
                    f"Please train the model first."
                )
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
    
    def predict(self, text):
        """
        Predict if text is spam
        
        Args:
            text: Input text message
            
        Returns:
            Tuple of (prediction, confidence)
            - prediction: 0 for ham, 1 for spam
            - confidence: Probability of the predicted class
        """
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False
        
        if self.model_type == 'distilbert':
            predictions, confidences = self._predict_bert(texts)
        else:
            predictions, confidences = self._predict_traditional(texts)
        
        if single:
            return int(predictions[0]), float(confidences[0])
        else:
            return predictions, confidences
    
    def _predict_traditional(self, texts):
        """Predict using traditional ML models"""
        # Vectorize
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence (probability of predicted class)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def _predict_bert(self, texts):
        """Predict using BERT model"""
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predictions and confidences
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        confidences = torch.max(probabilities, dim=1).values.cpu().numpy()
        
        return predictions, confidences
    
    def predict_with_details(self, text):
        """
        Predict with detailed information
        
        Args:
            text: Input text message
            
        Returns:
            Dictionary with prediction details
        """
        prediction, confidence = self.predict(text)
        
        # Get probabilities for both classes
        if self.model_type == 'distilbert':
            encodings = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        else:
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]
        
        return {
            'text': text,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'prediction_binary': int(prediction),
            'confidence': float(confidence),
            'probability_ham': float(probabilities[0]),
            'probability_spam': float(probabilities[1]),
            'model_used': self.model_type
        }


class EnsemblePredictor:
    """Ensemble predictor using multiple models"""
    
    def __init__(self, models=None):
        """
        Initialize ensemble predictor
        
        Args:
            models: List of model types to use. If None, uses all available models.
        """
        if models is None:
            models = ['naive_bayes', 'xgboost', 'svm']  # Exclude BERT by default (slower)
        
        self.predictors = {}
        for model_type in models:
            try:
                self.predictors[model_type] = SpamPredictor(model_type)
                logger.info(f"Loaded {model_type} for ensemble")
            except Exception as e:
                logger.warning(f"Could not load {model_type}: {e}")
        
        if not self.predictors:
            raise ValueError("No models could be loaded for ensemble")
        
        logger.info(f"Ensemble initialized with {len(self.predictors)} models")
    
    def predict(self, text, method='voting'):
        """
        Predict using ensemble
        
        Args:
            text: Input text message
            method: 'voting' (majority vote) or 'average' (average probabilities)
            
        Returns:
            Tuple of (prediction, confidence)
        """
        predictions = []
        confidences = []
        
        for model_type, predictor in self.predictors.items():
            pred, conf = predictor.predict(text)
            predictions.append(pred)
            confidences.append(conf)
        
        if method == 'voting':
            # Majority vote
            final_prediction = int(np.round(np.mean(predictions)))
            final_confidence = np.mean(confidences)
        else:  # average
            # Average probabilities
            final_prediction = 1 if np.mean(predictions) >= 0.5 else 0
            final_confidence = np.mean(confidences)
        
        return final_prediction, final_confidence
    
    def predict_with_details(self, text):
        """Get detailed predictions from all models"""
        results = {}
        
        for model_type, predictor in self.predictors.items():
            results[model_type] = predictor.predict_with_details(text)
        
        # Ensemble prediction
        predictions = [r['prediction_binary'] for r in results.values()]
        final_prediction = int(np.round(np.mean(predictions)))
        
        return {
            'text': text,
            'ensemble_prediction': 'spam' if final_prediction == 1 else 'ham',
            'ensemble_prediction_binary': final_prediction,
            'individual_predictions': results,
            'agreement': len(set(predictions)) == 1  # All models agree
        }


# Example usage
if __name__ == "__main__":
    # Test messages
    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "CONGRATULATIONS! You won $1000. Click here to claim your prize NOW!",
        "Can you send me the report by EOD?",
        "URGENT! Your account will be closed. Call this number immediately!",
        "Thanks for the update. I'll review it this afternoon."
    ]
    
    print("="*80)
    print("TESTING SPAM PREDICTOR")
    print("="*80)
    
    # Test single model
    print("\n1. Testing Naive Bayes Model:")
    print("-"*80)
    predictor = SpamPredictor('naive_bayes')
    
    for msg in test_messages:
        result = predictor.predict_with_details(msg)
        print(f"\nMessage: {msg}")
        print(f"Prediction: {result['prediction'].upper()} ({result['confidence']:.2%} confidence)")
        print(f"  Ham: {result['probability_ham']:.2%} | Spam: {result['probability_spam']:.2%}")
    
    # Test ensemble
    print("\n\n2. Testing Ensemble (Naive Bayes + XGBoost + SVM):")
    print("-"*80)
    
    try:
        ensemble = EnsemblePredictor()
        
        for msg in test_messages:
            result = ensemble.predict_with_details(msg)
            print(f"\nMessage: {msg}")
            print(f"Ensemble: {result['ensemble_prediction'].upper()}")
            print(f"Agreement: {'✓ All agree' if result['agreement'] else '✗ Disagreement'}")
            
            for model, pred in result['individual_predictions'].items():
                print(f"  {model:12s}: {pred['prediction']:4s} ({pred['confidence']:.2%})")
    
    except Exception as e:
        print(f"Ensemble test skipped: {e}")
    
    print("\n" + "="*80)
