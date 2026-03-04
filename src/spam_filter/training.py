from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

from spam_filter.datasets import load_dataset
from spam_filter.cleaning import clean_dataset
from spam_filter.models import create_model
from spam_filter.config import Config

from loguru import logger


def train(config: Config) -> tuple[Pipeline, pd.Series, pd.Series]:

    data_path = config.dataset.path
    test_size = config.split.test_size
    random_state = config.split.random_state
    max_iter = config.model.max_iter

    logger.info("Starting training pipeline")

    # Load dataset
    df = load_dataset(data_path)

    # Clean dataset
    df = clean_dataset(df)

    # Prepare features and labels
    X = df["message"]
    y = df["label"]

    logger.debug("Label distribution: {}", y.value_counts().to_dict())

    # Split dataset
    logger.info("Splitting dataset (test_size={})", test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.debug("Train samples: {}", len(X_train))
    logger.debug("Test samples: {}", len(X_test))

    # Create model
    model = create_model(max_iter)

    # Train model
    logger.info("Starting model training")
    model.fit(X_train, y_train)
    logger.info("Model training finished")

    return model, X_test, y_test