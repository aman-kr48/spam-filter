import pytest
import pandas as pd
from spam_filter.models import create_model

@pytest.fixture
def sample_dataset():
    X = pd.Series([
        "hello",
        "buy now",
        "hello friend",
        "cheap meds"
    ])

    y = pd.Series([0, 1, 0, 1])

    return X, y

@pytest.fixture
def trained_model(sample_dataset):

    X, y = sample_dataset

    model = create_model(100)
    model.fit(X, y)

    return model