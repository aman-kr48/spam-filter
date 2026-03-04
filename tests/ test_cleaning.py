import pandas as pd
import pytest

from spam_filter.cleaning import clean_dataset


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["ham", "spam"], {0, 1}),
        (["ham", "ham"], {0}),
        (["spam", "spam"], {1}),
    ],
)
def test_clean_dataset_converts_labels(labels, expected):

    df = pd.DataFrame({"label": labels, "message": ["msg"] * len(labels)})

    cleaned = clean_dataset(df)

    assert set(cleaned["label"]) == expected


def test_clean_dataset_removes_duplicates():

    df = pd.DataFrame({"label": ["ham", "ham"], "message": ["hello", "hello"]})

    cleaned = clean_dataset(df)

    assert len(cleaned) == 1
