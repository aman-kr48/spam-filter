def clean_dataset(df):

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    df = df.drop_duplicates()

    return df