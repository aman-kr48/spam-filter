import pandas as pd
from loguru import logger


def load_dataset(data_path: str) -> pd.DataFrame:

    logger.info("Loading dataset from {}", data_path)

    df = pd.read_csv(data_path, delimiter="\t", header=None, names=["label", "message"])

    logger.debug("Dataset loaded with shape {}", df.shape)

    return df
