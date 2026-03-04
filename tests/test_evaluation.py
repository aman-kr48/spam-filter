from spam_filter.evaluation import evaluate


def test_evaluate_returns_report(trained_model, sample_dataset):

    X, y = sample_dataset

    report = evaluate(trained_model, X, y)

    assert isinstance(report, str)
    assert "precision" in report
