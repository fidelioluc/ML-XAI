from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# --- TF-IDF ---
def apply_tfidf(docs):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=10)
    X = vectorizer.fit_transform(docs)
    return X, vectorizer


# --- Doc2Vec ---
def prepare_tagged_data(docs):
    """Tokenizes and tags documents for Doc2Vec training."""
    return [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(docs)]

def apply_doc2vec(docs, vector_size=100, epochs=20, window=5, min_count=2):
    tagged_data = prepare_tagged_data(docs)
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)

    # Inference: convert each doc to a vector
    vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in docs]
    return vectors, model


# --- S-BERT ---
def apply_sbert(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    vectors = model.encode(docs)
    return vectors, model
