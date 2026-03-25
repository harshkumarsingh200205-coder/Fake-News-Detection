# Preprocessing Pipeline

This document explains the text preprocessing implemented in `backend/preprocessing.py`.

## Purpose

The preprocessing stage converts raw article text into normalized tokens that are more suitable for TF-IDF vectorization and Logistic Regression.

## Resource Strategy

The preprocessor looks for local NLTK resources under `backend/nltk_data/`.

Resources checked:

- `punkt`
- `punkt_tab`
- `stopwords`
- `wordnet`
- `omw-1.4`

If some resources are missing, the code falls back gracefully instead of failing.

## Step-by-Step Pipeline

### 1. Raw Text Cleanup

`clean_text()` removes or normalizes:

- HTML tags
- URLs
- email addresses
- `@mentions`
- `#hashtags` into plain words
- repeated whitespace

### 2. Lowercasing

By default, all text is converted to lowercase so that words such as `News` and `news` map to the same token space.

### 3. Punctuation Handling

Punctuation is removed with `str.translate(...)` when `remove_punctuation=True`.

### 4. Number Handling

Numbers are preserved in the default configuration because `remove_numbers=False`.

### 5. Tokenization

The preprocessor prefers `nltk.word_tokenize(...)` when tokenization resources are available.

Fallback behavior:

- if NLTK tokenization resources are unavailable, it uses a regex tokenizer

### 6. Stopword Removal

The stopword set comes from:

- NLTK English stopwords when available
- otherwise sklearn's `ENGLISH_STOP_WORDS`

The code then adds custom news-related stopwords such as:

- `said`
- `told`
- `would`
- `reuters`
- `ap`
- `associated press`

### 7. Lemmatization

If WordNet is available, the preprocessor uses `WordNetLemmatizer` to reduce tokens to simpler base forms.

If WordNet is not available, the pipeline skips lemmatization and continues.

### 8. Minimum Token Length

Tokens shorter than `2` characters are removed in the default configuration.

## Output Format

The final output is a single whitespace-joined normalized string. That string is what the training and inference pipelines pass into TF-IDF.

## Current Default Configuration

`get_preprocessor()` returns:

- `remove_stopwords=True`
- `lemmatize=True`
- `lowercase=True`
- `remove_punctuation=True`
- `remove_numbers=False`
- `min_word_length=2`

## Why This Matters for the Model

This preprocessing reduces noise and helps the TF-IDF vectorizer focus on meaningful article terms and short phrases instead of formatting artifacts, URLs, punctuation, or overly common filler words.
