"""Regular tests without benchmarking for overhead comparison."""

import time
import numpy as np
from codspeed_trial import cpp_sum_of_squares, fibonacci


def test_cpp_sum_of_squares_small():
    """Test C++ sum of squares with small array (no benchmarking)."""
    arr = np.random.rand(100)
    start = time.perf_counter()
    result = cpp_sum_of_squares(arr)
    elapsed = time.perf_counter() - start
    print(f"\nSmall array (100 elements): {elapsed*1e6:.2f} µs")
    assert isinstance(result, float)


def test_cpp_sum_of_squares_medium():
    """Test C++ sum of squares with medium array (no benchmarking)."""
    arr = np.random.rand(10_000)
    start = time.perf_counter()
    result = cpp_sum_of_squares(arr)
    elapsed = time.perf_counter() - start
    print(f"\nMedium array (10k elements): {elapsed*1e6:.2f} µs")
    assert isinstance(result, float)


def test_cpp_sum_of_squares_large():
    """Test C++ sum of squares with large array (no benchmarking)."""
    arr = np.random.rand(1_000_000)
    start = time.perf_counter()
    result = cpp_sum_of_squares(arr)
    elapsed = time.perf_counter() - start
    print(f"\nLarge array (1M elements): {elapsed*1e3:.2f} ms")
    assert isinstance(result, float)


def test_fibonacci_small():
    """Test Cython fibonacci with small input (no benchmarking)."""
    start = time.perf_counter()
    result = fibonacci(20)
    elapsed = time.perf_counter() - start
    print(f"\nFibonacci(20): {elapsed*1e6:.2f} µs")
    assert result == 6765


def test_fibonacci_medium():
    """Test Cython fibonacci with medium input (no benchmarking)."""
    start = time.perf_counter()
    result = fibonacci(100)
    elapsed = time.perf_counter() - start
    print(f"\nFibonacci(100): {elapsed*1e6:.2f} µs")
    assert isinstance(result, int)


def test_fibonacci_large():
    """Test Cython fibonacci with large input (no benchmarking)."""
    start = time.perf_counter()
    result = fibonacci(1000)
    elapsed = time.perf_counter() - start
    print(f"\nFibonacci(1000): {elapsed*1e6:.2f} µs")
    assert isinstance(result, int)


def test_mixed_workload():
    """Test mixed C++ and Cython operations (no benchmarking)."""
    arr = np.random.rand(50_000)
    start = time.perf_counter()
    result_cpp = cpp_sum_of_squares(arr)
    result_cy = fibonacci(50)
    result = result_cpp + result_cy
    elapsed = time.perf_counter() - start
    print(f"\nMixed workload: {elapsed*1e6:.2f} µs")
    assert isinstance(result, float)
