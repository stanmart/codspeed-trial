from time import perf_counter

import pytest

from codspeed_trial import create_logistic_dataset, fit_logistic_regression


@pytest.fixture
def logistic_data():
    return create_logistic_dataset(n=5_000_000, p=100)


def test_fit_logistic_regression(logistic_data):
    X, y = logistic_data

    start = perf_counter()
    model = fit_logistic_regression(X, y)
    elapsed = perf_counter() - start
    print(f"Elapsed time: {elapsed:.2f} seconds")

    assert hasattr(model, "coef_")
    assert len(model.coef_) == X.shape[1]
