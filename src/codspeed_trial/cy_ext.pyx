# cython: language_level=3
"""Cython extension module with simple computational functions."""

def fibonacci(int n):
    """
    Compute the nth Fibonacci number with loop unrolling optimization.

    Parameters
    ----------
    n : int
        The index of the Fibonacci number to compute (must be >= 0).

    Returns
    -------
    int
        The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    cdef long long a = 0
    cdef long long b = 1
    cdef long long temp
    cdef int i
    cdef int num_iters = n - 1  # Number of iterations needed
    cdef int unrolled_iters = (num_iters // 4) * 4  # Round down to multiple of 4

    # Loop unrolling: process 4 iterations at a time
    i = 0
    while i < unrolled_iters:
        # Iteration 1
        temp = a + b
        a = b
        b = temp
        # Iteration 2
        temp = a + b
        a = b
        b = temp
        # Iteration 3
        temp = a + b
        a = b
        b = temp
        # Iteration 4
        temp = a + b
        a = b
        b = temp
        i += 4

    # Handle remaining iterations
    while i < num_iters:
        temp = a + b
        a = b
        b = temp
        i += 1

    return b


cpdef long long fibonacci_fast(int n) nogil:
    """
    Fast Fibonacci computation with GIL released.

    Parameters
    ----------
    n : int
        The index of the Fibonacci number to compute.

    Returns
    -------
    long long
        The nth Fibonacci number.
    """
    if n <= 1:
        return n

    cdef long long a = 0
    cdef long long b = 1
    cdef long long temp
    cdef int i

    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp

    return b
