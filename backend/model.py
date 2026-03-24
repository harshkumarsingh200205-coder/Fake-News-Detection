# TF-IDF + Logistic Regression
"""
Machine Learning Model Module for Fake News Detection
Implements TF-IDF Vectorizer + Logistic Regression pipeline
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json

# Default paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.joblib')
METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')


class FakeNewsModel:
    """Fake News Detection Model using TF-IDF + Logistic Regression"""
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 random_state: int = 42):
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True
        )

        self.model = self._create_classifier()
        
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.is_fitted = False
        self.metrics = {}
        self.feature_names = []

    def _create_classifier(self, **overrides: Any) -> LogisticRegression:
        """Create a classifier compatible with the installed scikit-learn."""
        params: Dict[str, Any] = {
            'C': self.C,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
        }

        # `multi_class` was removed from newer scikit-learn releases.
        if 'multi_class' in LogisticRegression().get_params(deep=False):
            params['multi_class'] = 'auto'

        params.update(overrides)
        return LogisticRegression(**params)
    
    def fit(self, X: List[str], y: List[int]) -> 'FakeNewsModel':
        """Train the model"""
        print(f"Training model on {len(X)} samples...")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        print("Model training complete!")
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline.predict_proba(X)
    
    def get_confidence(self, X: List[str]) -> List[Tuple[str, float]]:
        """Get prediction with confidence score"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            confidence = max(probs)
            label = "REAL" if pred == 1 else "FAKE"
            results.append((label, confidence))
        
        return results
    
    def get_keyword_importance(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get influential keywords for a prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting keywords")
        
        tfidf_vector = self.vectorizer.transform([text])
        coefficients = self.model.coef_[0]
        non_zero_indices = tfidf_vector.nonzero()[1]
        
        keyword_scores = []
        for idx in non_zero_indices:
            word = self.feature_names[idx]
            tfidf_score = tfidf_vector[0, idx]
            importance = tfidf_score * coefficients[idx]
            keyword_scores.append((word, importance))
        
        keyword_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return keyword_scores[:top_n]
    
    def get_top_fake_keywords(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top keywords that indicate fake news"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coefficients = self.model.coef_[0]
        fake_indices = np.argsort(coefficients)[:top_n]
        
        return [(self.feature_names[idx], coefficients[idx]) for idx in fake_indices]
    
    def get_top_real_keywords(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top keywords that indicate real news"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coefficients = self.model.coef_[0]
        real_indices = np.argsort(coefficients)[-top_n:][::-1]
        
        return [(self.feature_names[idx], coefficients[idx]) for idx in real_indices]
    
    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict[str, Any]:
        """Evaluate the model"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
        }
        
        self.metrics = metrics
        return metrics
    
    def cross_validate(self, X: List[str], y: List[int], cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def save(self, model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH) -> None:
        """Save the model and metrics"""
        joblib.dump(self.pipeline, model_path, protocol=4)
        print(f"Model saved to {model_path}")
        
        if self.metrics:
            metrics_to_save = {}
            for key, value in self.metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_to_save[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    metrics_to_save[key] = float(value)
                else:
                    metrics_to_save[key] = value
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"Metrics saved to {metrics_path}")
    
    def load(self, model_path: str = MODEL_PATH) -> 'FakeNewsModel':
        """Load a trained model"""
        self.pipeline = joblib.load(model_path)
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.model = self.pipeline.named_steps['classifier']

        self.is_fitted = (
            hasattr(self.vectorizer, 'vocabulary_')
            and hasattr(self.model, 'classes_')
            and hasattr(self.model, 'coef_')
        )
        self.feature_names = (
            list(self.vectorizer.get_feature_names_out())
            if self.is_fitted and hasattr(self.vectorizer, 'get_feature_names_out')
            else []
        )
        print(f"Model loaded from {model_path}")
        return self
    
    def retrain_with_new_data(self, new_texts: List[str], new_labels: List[int], 
                              incremental: bool = True) -> 'FakeNewsModel':
        """Retrain model with new data, optionally incrementally"""
        if incremental and self.is_fitted:
            # For incremental learning, we need to retrain on combined data
            # This is a simplified approach - in production you'd use partial_fit
            print(f"Retraining model incrementally with {len(new_texts)} new samples...")
            # For now, we'll retrain from scratch with new data
            # In a real scenario, you'd want to store historical data
            self.pipeline.fit(new_texts, new_labels)
        else:
            print(f"Training new model with {len(new_texts)} samples...")
            self.pipeline.fit(new_texts, new_labels)
        
        self.is_fitted = True
        self.feature_names = self.vectorizer.get_feature_names_out()
        print("Model retraining complete!")
        return self


def get_model() -> FakeNewsModel:
    """Factory function to get the model instance"""
    model = FakeNewsModel()

    if os.path.exists(MODEL_PATH):
        try:
            model.load()
            if model.is_fitted:
                print("Loaded existing trained model")
            else:
                print("Loaded model file is not fitted. Resetting model instance.")
                model = FakeNewsModel()
        except Exception as e:
            print(f"Could not load model: {e}. Removing old model and resetting.")
            try:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
            except Exception as delete_err:
                print(f"Could not remove invalid model file: {delete_err}")
            model = FakeNewsModel()

    return model
