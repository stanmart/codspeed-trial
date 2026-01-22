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

    // Compute sum of squares
    double result = 0.0;
    for (npy_intp i = 0; i < size; i++) {
        result += data[i] * data[i];
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
