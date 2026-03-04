import omegaconf as oc

from spam_filter.scripts import main


def test_main_runs_and_prints_output(tmp_path, capsys):

    data_file = tmp_path / "sms.txt"

    data_file.write_text("ham\tHello\nspam\tBuy now\nham\tHi there\nspam\tCheap meds\n")

    config_file = tmp_path / "config.yaml"

    config_file.write_text(f"""
dataset:
  path: {data_file}

split:
  test_size: 0.5
  random_state: 42

model:
  max_iter: 100
""")

    result = main([str(config_file)])

    captured = capsys.readouterr()

    assert result == 0
    assert "Training completed." in captured.out
