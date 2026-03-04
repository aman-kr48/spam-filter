from sklearn.pipeline import Pipeline

from spam_filter.models import create_model


def test_create_model_returns_pipeline():

    model = create_model(100)

    assert isinstance(model, Pipeline)


def test_create_model_pipeline_steps():

    model = create_model(100)

    step_names = [name for name, _ in model.steps]

    assert "vectorizer" in step_names
    assert "model" in step_names
