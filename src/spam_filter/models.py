from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from spam_filter.config import MAX_ITER


def create_model():

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=MAX_ITER))
    ])

    return model