"""CodSpeed benchmark tests."""

import numpy as np
import pytest
from codspeed_trial import cpp_sum_of_squares, fibonacci


@pytest.mark.benchmark
def test_cpp_sum_of_squares_small(benchmark):
    """Benchmark C++ sum of squares with small array."""
    arr = np.random.rand(100)

    @benchmark
    def run():
        return cpp_sum_of_squares(arr)


@pytest.mark.benchmark
def test_cpp_sum_of_squares_medium(benchmark):
    """Benchmark C++ sum of squares with medium array."""
    arr = np.random.rand(10_000)

    @benchmark
    def run():
        return cpp_sum_of_squares(arr)


@pytest.mark.benchmark
def test_cpp_sum_of_squares_large(benchmark):
    """Benchmark C++ sum of squares with large array."""
    arr = np.random.rand(1_000_000)

    @benchmark
    def run():
        return cpp_sum_of_squares(arr)


@pytest.mark.benchmark
def test_fibonacci_small(benchmark):
    """Benchmark Cython fibonacci with small input."""

    @benchmark
    def run():
        return fibonacci(20)


@pytest.mark.benchmark
def test_fibonacci_medium(benchmark):
    """Benchmark Cython fibonacci with medium input."""

    @benchmark
    def run():
        return fibonacci(100)


@pytest.mark.benchmark
def test_fibonacci_large(benchmark):
    """Benchmark Cython fibonacci with large input."""

    @benchmark
    def run():
        return fibonacci(1000)


@pytest.mark.benchmark
def test_mixed_workload(benchmark):
    """Benchmark mixed C++ and Cython operations."""
    arr = np.random.rand(50_000)

    @benchmark
    def run():
        result_cpp = cpp_sum_of_squares(arr)
        result_cy = fibonacci(50)
        return result_cpp + result_cy
