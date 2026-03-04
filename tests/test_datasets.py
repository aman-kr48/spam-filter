import pandas as pd
from spam_filter.datasets import load_dataset


def test_load_dataset_reads_file(tmp_path):

    data_file = tmp_path / "sms.txt"

    data_file.write_text(
        "ham\tHello\n"
        "spam\tBuy now\n"
    )

    df = load_dataset(str(data_file))

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["label", "message"]
    assert len(df) == 2