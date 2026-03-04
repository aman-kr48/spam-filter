import omegaconf as oc
import pandas as pd
import pytest

from spam_filter.training import train


def test_train_returns_model(tmp_path):

    data_file = tmp_path / "sms.txt"

    data_file.write_text("ham\tHello\nspam\tBuy now\nham\tHi there\nspam\tCheap meds\n")

    config = oc.OmegaConf.create(
        {
            "dataset": {"path": str(data_file)},
            "split": {"test_size": 0.5, "random_state": 42},
            "model": {"max_iter": 100},
        }
    )

    model, X_test, y_test = train(config)

    assert len(X_test) > 0
    assert len(y_test) > 0


def test_train_raises_error_on_invalid_path():

    config = oc.OmegaConf.create(
        {
            "dataset": {"path": "non_existent.txt"},
            "split": {"test_size": 0.5, "random_state": 42},
            "model": {"max_iter": 100},
        }
    )

    with pytest.raises(Exception):
        train(config)


def test_train_split_is_reproducible(tmp_path):

    data_file = tmp_path / "sms.txt"

    data_file.write_text(
        "ham\tHello\nspam\tBuy now\nham\tHi there\nspam\tCheap meds\nham\tMorning\nspam\tOffer\n"
    )

    config = oc.OmegaConf.create(
        {
            "dataset": {"path": str(data_file)},
            "split": {"test_size": 0.5, "random_state": 42},
            "model": {"max_iter": 100},
        }
    )

    _, X_test_1, _ = train(config)
    _, X_test_2, _ = train(config)

    assert list(X_test_1) == list(X_test_2)


def test_training_fails_on_empty_dataset(tmp_path):

    data_file = tmp_path / "sms.txt"
    data_file.write_text("")

    config = oc.OmegaConf.create(
        {
            "dataset": {"path": str(data_file)},
            "split": {"test_size": 0.5, "random_state": 42},
            "model": {"max_iter": 100},
        }
    )

    with pytest.raises(Exception):
        train(config)
