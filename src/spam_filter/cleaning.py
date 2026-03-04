import pandas as pd
from loguru import logger


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Cleaning dataset")

    initial_rows = len(df)

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    df = df.drop_duplicates()

    removed_rows = initial_rows - len(df)

    logger.debug("Removed {} duplicate rows", removed_rows)

    logger.debug("Label distribution after cleaning: {}", df["label"].value_counts().to_dict())

    return df
