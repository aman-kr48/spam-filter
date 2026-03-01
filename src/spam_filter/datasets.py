import pandas as pd


def load_dataset(data_path):

    df = pd.read_csv(
        data_path,
        delimiter="\t",
        header=None,
        names=["label", "message"]
    )

    return df