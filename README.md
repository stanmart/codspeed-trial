# codspeed-trial

Test repository for evaluating CodSpeed continuous benchmarking service.

## Overview

This package includes:
- **C++ extension**: NumPy-interfaced module for computing sum of squares
- **Cython extension**: Optimized Fibonacci number computation
- **CodSpeed benchmarks**: Performance tests using `pytest-codspeed`
- **Overhead tests**: Identical tests without benchmarking for comparison

## Setup

Install dependencies with [pixi](https://pixi.sh):

```bash
pixi install
```

Build the package:

```bash
pixi run build
```

## Running Tests

Run CodSpeed benchmarks:

```bash
pixi run benchmark
```

Run overhead comparison tests:

```bash
pixi run test-overhead
```

Run all tests:

```bash
pixi run test
```

## CI/CD

The GitHub Actions workflow (`.github/workflows/codspeed.yml`) automatically:
- Builds the package on `ubuntu-latest` runners
- Runs benchmarks with CodSpeed on pushes and PRs
- Requires `CODSPEED_TOKEN` secret to be set in repository settings

## Project Structure

```
src/codspeed_trial/
  ├── __init__.py
  ├── cpp_ext.cpp          # C++ extension with NumPy
  └── cy_ext.pyx           # Cython extension
tests/
  ├── test_benchmarks.py   # CodSpeed benchmark suite
  └── test_overhead.py     # Non-benchmarked tests for overhead comparison
```

## Extensions

- **cpp_ext.sum_of_squares**: Computes Σx² for 1D float64 arrays
- **cy_ext.fibonacci**: Computes nth Fibonacci number with Cython optimization
