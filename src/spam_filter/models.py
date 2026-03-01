from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def create_model(max_iter):

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=max_iter))
    ])

    return model