"""CodSpeed trial package for benchmarking evaluation."""

__version__ = "0.1.0"

# Import extensions when they're built
try:
    from .cpp_ext import sum_of_squares as cpp_sum_of_squares
except ImportError:
    cpp_sum_of_squares = None

try:
    from .cy_ext import fibonacci
except ImportError:
    fibonacci = None

from .glum_thing import create_logistic_dataset, fit_logistic_regression

__all__ = [
    "cpp_sum_of_squares",
    "fibonacci",
    "create_logistic_dataset",
    "fit_logistic_regression",
]