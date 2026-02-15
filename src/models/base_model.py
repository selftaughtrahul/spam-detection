"""
Base Model Class
Provides common interface for all classification models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path


class BaseSpamClassifier(ABC):
    """Abstract base class for spam classifiers"""
    
    def __init__(self):
        self.is_trained = False
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: Path):
        """
        Save model to disk
        
        Args:
            filepath: Path to save file
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: Path):
        """
        Load model from disk
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded model instance
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.model_name}(trained={self.is_trained})"
