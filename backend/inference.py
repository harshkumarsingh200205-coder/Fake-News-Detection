# Prediction logic
"""
Inference Module for Fake News Detection
Handles predictions, URL scraping, and keyword extraction
"""

import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import requests

from model import FakeNewsModel, get_model, MODEL_PATH
from preprocessing import TextPreprocessor, get_preprocessor

# Model and preprocessor instances
_model: Optional[FakeNewsModel] = None
_preprocessor: Optional[TextPreprocessor] = None


def initialize():
    """Initialize model and preprocessor"""
    global _model, _preprocessor
    
    if _model is None:
        _model = get_model()
    
    if _preprocessor is None:
        _preprocessor = get_preprocessor()
    
    return _model, _preprocessor


def get_instances() -> Tuple[FakeNewsModel, TextPreprocessor]:
    """Get model and preprocessor instances"""
    global _model, _preprocessor

    if _model is None or _preprocessor is None:
        initialize()

    # If a backend process started before training completed, reload the model
    # as soon as a saved artifact becomes available.
    if _model is not None and not _model.is_fitted and os.path.exists(MODEL_PATH):
        try:
            _model = get_model()
        except Exception as e:
            print(f"Model reload skipped: {e}")

    return _model, _preprocessor


def replace_model(model: FakeNewsModel) -> FakeNewsModel:
    """Replace the cached in-memory model after retraining."""
    global _model
    _model = model
    return _model


class URLScraper:
    """Web scraper for extracting news content from URLs"""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    TIMEOUT = 10
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        pattern = re.compile(
            r'^https?://' r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|' r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(pattern.match(url))
    
    @staticmethod
    def fetch_content(url: str) -> Optional[str]:
        """Fetch HTML content from URL"""
        try:
            response = requests.get(url, headers=URLScraper.HEADERS, 
                                   timeout=URLScraper.TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
    
    @staticmethod
    def extract_text(html: str) -> str:
        """Extract main text content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        selectors = ['article', '[class*="article"]', '[class*="content"]', 
                    '[class*="post"]', 'main', '.entry-content']
        
        main_content = None
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                main_content = content
                break
        
        if main_content is None:
            main_content = soup.body if soup.body else soup
        
        paragraphs = main_content.find_all('p')
        if paragraphs:
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        else:
            text = main_content.get_text(separator=' ', strip=True)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @classmethod
    def scrape_url(cls, url: str) -> Dict[str, Any]:
        """Scrape content from a URL"""
        if not cls.is_valid_url(url):
            return {'success': False, 'error': 'Invalid URL format', 'text': None}
        
        html = cls.fetch_content(url)
        if html is None:
            return {'success': False, 'error': 'Failed to fetch URL', 'text': None}
        
        text = cls.extract_text(html)
        if not text or len(text) < 50:
            return {'success': False, 'error': 'Could not extract sufficient text', 'text': text}
        
        return {'success': True, 'error': None, 'text': text, 'text_length': len(text)}


class FakeNewsPredictor:
    """Main prediction class"""

    FAKE_INDICATORS = (
        'shocking',
        "you won't believe",
        'miracle',
        'secret',
        'conspiracy',
        'breaking',
        'unbelievable',
        'must see',
        'exposed',
        'viral',
    )
    REAL_INDICATORS = (
        'according to',
        'reported',
        'official',
        'statement',
        'research',
        'study',
        'published',
        'confirmed',
        'announced',
        'data',
    )
    
    def __init__(self):
        self.model, self.preprocessor = get_instances()
    
    def predict(self, text: str, return_keywords: bool = True, 
                top_keywords: int = 10) -> Dict[str, Any]:
        """Predict whether news is fake or real"""
        start_time = time.time()
        
        processed_text = self.preprocessor.preprocess(text)
        
        if not processed_text or len(processed_text.split()) < 3:
            return {
                'success': False,
                'error': 'Text too short after preprocessing',
                'prediction': None,
                'confidence': None
            }
        
        if not self.model.is_fitted:
            return self._mock_prediction(text, processed_text, top_keywords, start_time)
        
        try:
            probabilities = self.model.predict_proba([processed_text])[0]
        except Exception as e:
            print(f"Prediction fallback activated: {e}")
            self.model.is_fitted = False
            return self._mock_prediction(text, processed_text, top_keywords, start_time)
        prediction = int(probabilities[1] > 0.5)
        confidence = float(max(probabilities))
        
        result = {
            'success': True,
            'error': None,
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': round(confidence * 100, 2),
            'fake_probability': round(float(probabilities[0]) * 100, 2),
            'real_probability': round(float(probabilities[1]) * 100, 2),
            'processing_time': round(time.time() - start_time, 3)
        }
        
        if return_keywords:
            try:
                keywords = self.model.get_keyword_importance(processed_text, top_keywords)
                result['keywords'] = [
                    {'word': word, 'importance': round(abs(score), 4), 
                     'type': 'fake' if score < 0 else 'real'}
                    for word, score in keywords
                ]
            except Exception as e:
                print(f"Keyword extraction fallback activated: {e}")
                result['keywords'] = self._build_demo_keywords(
                    raw_text=text,
                    processed_text=processed_text,
                    prediction=result['prediction'],
                    top_keywords=top_keywords,
                )
        
        return result
    
    def _build_demo_keywords(
        self,
        raw_text: str,
        processed_text: str,
        prediction: str,
        top_keywords: int,
    ) -> List[Dict[str, Any]]:
        """Generate heuristic keywords so the explanation UI remains useful."""
        text_lower = raw_text.lower()
        token_counts = Counter(
            token for token in processed_text.split() if len(token) >= 3
        )
        signed_scores: Dict[str, float] = {}

        for phrase in self.FAKE_INDICATORS:
            if phrase in text_lower:
                signed_scores[phrase] = signed_scores.get(phrase, 0.0) - 1.5

        for phrase in self.REAL_INDICATORS:
            if phrase in text_lower:
                signed_scores[phrase] = signed_scores.get(phrase, 0.0) + 1.5

        default_direction = 1.0 if prediction == 'REAL' else -1.0
        total_tokens = max(sum(token_counts.values()), 1)

        for token, count in token_counts.most_common(top_keywords * 3):
            if token in signed_scores:
                continue
            signed_scores[token] = default_direction * (count / total_tokens)

        ranked_keywords = sorted(
            signed_scores.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_keywords]

        return [
            {
                'word': word,
                'importance': round(abs(score), 4),
                'type': 'fake' if score < 0 else 'real',
            }
            for word, score in ranked_keywords
        ]

    def _mock_prediction(
        self,
        raw_text: str,
        processed_text: str,
        top_keywords: int,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate mock prediction when model is not trained"""
        text_lower = raw_text.lower()
        fake_count = sum(1 for i in self.FAKE_INDICATORS if i in text_lower)
        real_count = sum(1 for i in self.REAL_INDICATORS if i in text_lower)
        
        fake_prob = 0.4 + (fake_count * 0.1) - (real_count * 0.1)
        fake_prob = max(0.1, min(0.9, fake_prob))
        real_prob = 1 - fake_prob
        
        prediction = 'REAL' if real_prob > fake_prob else 'FAKE'
        confidence = max(fake_prob, real_prob) * 100
        processing_time = round(time.time() - start_time, 3) if start_time else None
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2),
            'keywords': self._build_demo_keywords(
                raw_text=raw_text,
                processed_text=processed_text,
                prediction=prediction,
                top_keywords=top_keywords,
            ),
            'processing_time': processing_time,
            'note': 'Demo mode - trained model not available'
        }
    
    def predict_from_url(self, url: str) -> Dict[str, Any]:
        """Predict from URL by scraping content"""
        scrape_result = URLScraper.scrape_url(url)
        
        if not scrape_result['success']:
            return {
                'success': False,
                'error': scrape_result['error'],
                'url': url,
                'prediction': None,
                'confidence': None
            }
        
        prediction_result = self.predict(scrape_result['text'])
        prediction_result['url'] = url
        prediction_result['source_text'] = scrape_result['text']
        prediction_result['scraped_text_length'] = scrape_result.get('text_length', 0)
        
        return prediction_result
