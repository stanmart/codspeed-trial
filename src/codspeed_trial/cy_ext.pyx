# cython: language_level=3
"""Cython extension module with simple computational functions."""

def fibonacci(int n):
    """
    Compute the nth Fibonacci number.

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

    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp

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
