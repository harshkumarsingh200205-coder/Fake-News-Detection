# Text preprocessing
"""
Text Preprocessing Module for Fake News Detection
Handles stopword removal, lemmatization, and text cleaning
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Optional
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_PATH)

RESOURCE_PATHS = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
}


def has_nltk_resource(resource: str) -> bool:
    """Check whether an NLTK resource exists locally, including zipped bundles."""
    path = RESOURCE_PATHS[resource]

    try:
        nltk.data.find(path)
        return True
    except LookupError:
        try:
            nltk.data.find(f'{path}.zip')
            return True
        except LookupError:
            return False


AVAILABLE_RESOURCES = {
    resource: has_nltk_resource(resource)
    for resource in RESOURCE_PATHS
}

missing_resources = [
    resource for resource, available in AVAILABLE_RESOURCES.items() if not available
]
if missing_resources:
    print(
        "Using local/fallback NLP resources. Missing optional NLTK packages: "
        + ", ".join(missing_resources)
    )


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for fake news detection.
    Implements stopword removal, lemmatization, and text cleaning.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2):
        
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        
        self.lemmatizer = (
            WordNetLemmatizer() if AVAILABLE_RESOURCES['wordnet'] else None
        )
        try:
            self.stop_words = (
                set(stopwords.words('english'))
                if AVAILABLE_RESOURCES['stopwords']
                else set(ENGLISH_STOP_WORDS)
            )
        except LookupError:
            self.stop_words = set(ENGLISH_STOP_WORDS)
        
        # Custom stopwords for news articles
        custom_stopwords = {
            'said', 'say', 'says', 'told', 'would', 'could', 'also',
            'one', 'two', 'three', 'like', 'get', 'make', 'go', 'know',
            'reuters', 'ap', 'associated press'
        }
        self.stop_words.update(custom_stopwords)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with NLTK when available, otherwise use a regex fallback."""
        try:
            if AVAILABLE_RESOURCES['punkt'] or AVAILABLE_RESOURCES['punkt_tab']:
                return word_tokenize(text)
        except LookupError:
            pass

        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
    
    def clean_text(self, text: str) -> str:
        """Clean raw text by removing HTML, URLs, and special characters"""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
        text = re.sub(r'http\S+|www\.\S+|https\S+', ' ', text, flags=re.MULTILINE)  # URLs
        text = re.sub(r'\S+@\S+', ' ', text)  # Emails
        text = re.sub(r'@\w+', ' ', text)  # Mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Hashtags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text
    
    def preprocess(self, text: str) -> str:
        """Apply full preprocessing pipeline to text"""
        text = self.clean_text(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        tokens = self.tokenize(text)
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        if self.lemmatize and self.lemmatizer is not None:
            try:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            except LookupError:
                pass
        
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        
        return ' '.join(tokens)


def get_preprocessor() -> TextPreprocessor:
    """Factory function to get a configured preprocessor"""
    return TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        min_word_length=2
    )
