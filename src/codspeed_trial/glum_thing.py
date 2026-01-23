import glum
import numpy as np
import pandas as pd


def create_logistic_dataset(n: int = 1_000_000, p: int = 10):
    # Generate random data
    X = pd.DataFrame({f"x__{i}": np.random.rand(n) for i in range(p)})
    y = np.random.randint(0, 2, size=n)
    return X, y


def fit_logistic_regression(X: pd.DataFrame, y: np.ndarray):
    model = glum.GeneralizedLinearRegressor(family="binomial", link="logit", alpha=0.1)
    model.fit(X, y)
    return model
