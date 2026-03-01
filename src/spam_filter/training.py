from sklearn.model_selection import train_test_split

from spam_filter.datasets import load_dataset
from spam_filter.cleaning import clean_dataset
from spam_filter.models import create_model

def train(config):

    data_path = config.dataset.path
    test_size = config.split.test_size
    random_state = config.split.random_state
    max_iter = config.model.max_iter

    df = load_dataset(data_path)

    df = clean_dataset(df)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = create_model(max_iter)

    model.fit(X_train, y_train)

    return model, X_test, y_test