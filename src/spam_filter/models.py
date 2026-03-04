from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from loguru import logger


def create_model(max_iter: int) -> Pipeline:

    logger.info("Creating LogisticRegression model with max_iter={}", max_iter)

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=max_iter))
    ])

    logger.debug("Model pipeline created: {}", model)

    return model