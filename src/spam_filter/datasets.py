import pandas as pd
from spam_filter.config import DATA_PATH, SEPARATOR, COLUMN_NAMES


def load_dataset():

    df = pd.read_csv(
        DATA_PATH,
        delimiter=SEPARATOR,
        header=None,
        names=COLUMN_NAMES
    )

    return df