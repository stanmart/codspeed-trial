import pytest

from codspeed_trial import create_logistic_dataset, fit_logistic_regression


@pytest.fixture
def logistic_data():
    return create_logistic_dataset(n=5_000_000, p=100)


@pytest.mark.benchmark
def test_fit_logistic_regression(logistic_data, benchmark):
    X, y = logistic_data

    model = benchmark(fit_logistic_regression, X, y)

    assert hasattr(model, "coef_")
    assert len(model.coef_) == X.shape[1]
