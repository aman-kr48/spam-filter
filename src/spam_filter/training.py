from sklearn.model_selection import train_test_split

from spam_filter.datasets import load_dataset
from spam_filter.cleaning import clean_dataset
from spam_filter.models import create_model
from spam_filter.config import TEXT_COLUMN, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def train():

    df = load_dataset()

    df = clean_dataset(df)

    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = create_model()

    model.fit(X_train, y_train)

    return model, X_test, y_test