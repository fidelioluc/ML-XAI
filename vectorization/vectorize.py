from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# --- Doc2Vec ---
class GensimDoc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=40):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        tagged_data = [
            TaggedDocument(words=x.split(), tags=[i]) for i, x in enumerate(X)
        ]
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
        )
        self.model.build_vocab(tagged_data)
        self.model.train(
            tagged_data, total_examples=len(tagged_data), epochs=self.epochs
        )
        return self

    def transform(self, X):
        return [self.model.infer_vector(x.split()) for x in X]


# --- S-BERT ---
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


class SBERTVectorizer(BaseEstimator, TransformerMixin):
    """SBERT Sentence Transformer Vectorizer"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Initialize the SBERT model
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to list
        texts = X.tolist() if hasattr(X, 'tolist') else list(X)

        # Replace invalid entries with empty strings and cast everything to str
        cleaned_texts = []
        replaced_count = 0
        for text in texts:
            if isinstance(text, str):
                cleaned_texts.append(text)
            else:
                cleaned_texts.append("")
                replaced_count += 1

        if replaced_count > 0:
            print(f"[Warning] Replaced {replaced_count} invalid text entries with empty strings.")

        # Generate embeddings
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=True)
        return embeddings

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

