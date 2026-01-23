#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cmath>

// Compute sum of squares of array elements
static PyObject* sum_of_squares(PyObject* self, PyObject* args) {
    PyArrayObject* array;

    // Parse input array
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }

    // Check that it's a 1D double array
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }

    if (PyArray_TYPE(array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Array must be of type float64");
        return NULL;
    }

    // Get array data and size
    double* data = (double*)PyArray_DATA(array);
    npy_intp size = PyArray_SIZE(array);

    // Compute sum of squares with loop unrolling
    // Optimization: Removed unnecessary trig computation (sin²+cos²=1)
    double result = 0.0;
    npy_intp i = 0;

    // Process 4 elements at a time (loop unrolling)
    for (; i + 3 < size; i += 4) {
        double val0 = data[i];
        double val1 = data[i + 1];
        double val2 = data[i + 2];
        double val3 = data[i + 3];
        result += val0 * val0 + val1 * val1 + val2 * val2 + val3 * val3;
    }

    // Handle remaining elements
    for (; i < size; i++) {
        double val = data[i];
        result += val * val;
    }

    return PyFloat_FromDouble(result);
}

// Method definitions
static PyMethodDef CppExtMethods[] = {
    {"sum_of_squares", sum_of_squares, METH_VARARGS,
     "Compute sum of squares of array elements"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cpp_ext_module = {
    PyModuleDef_HEAD_INIT,
    "cpp_ext",
    "C++ extension module with NumPy interface",
    -1,
    CppExtMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_cpp_ext(void) {
    import_array();
    return PyModule_Create(&cpp_ext_module);
}
