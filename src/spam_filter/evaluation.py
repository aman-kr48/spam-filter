import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def evaluate(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> str:

    logger.info("Evaluating model")
    predictions = model.predict(X_test)
    logger.debug("Prediction completed for {} samples", len(X_test))

    report = classification_report(y_test, predictions)
    logger.info("Evaluation finished")

    return report
