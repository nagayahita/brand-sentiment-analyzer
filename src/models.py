from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

class SentimentAnalyzer:
    """Sentiment analysis using multiple models."""
    
    def __init__(self):
        """Initialize models and vectorizer."""
        # Setup vectorizer and models
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            'logistic': LogisticRegression(max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.trained_models = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train(self, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """
        Train all models and return performance metrics.
        
        Args:
            X (pd.Series): Preprocessed text data
            y (pd.Series): Target labels
            
        Returns:
            Dict[str, Any]: Performance metrics for each model
        """
        try:
            # Transform text data
            X_vectorized = self.vectorizer.fit_transform(X)
            
            metrics = {}
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                model.fit(X_vectorized, y)
                self.trained_models[name] = model
                
                # Calculate metrics
                predictions = model.predict(X_vectorized)
                metrics[name] = {
                    'classification_report': classification_report(y, predictions),
                    'confusion_matrix': confusion_matrix(y, predictions)
                }
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, text: str, model_name: str = 'logistic') -> int:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text
            model_name (str): Name of model to use
            
        Returns:
            int: Predicted sentiment (-1, 0, or 1)
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
                
            # Vectorize text
            X_vec = self.vectorizer.transform([text])
            
            # Predict
            return self.trained_models[model_name].predict(X_vec)[0]
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise
