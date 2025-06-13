from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


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
class SbertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return self.model.encode(X, show_progress_bar=False)
