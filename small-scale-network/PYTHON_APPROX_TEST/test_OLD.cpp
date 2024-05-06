#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>

// Your function here
PyObject* conv2d_approx(int i, PyArrayObject* inputs, PyArrayObject* kernel) {
    std::cout << "Hej";
    npy_intp output_dims[3] = {PyArray_DIM(inputs, 1) - PyArray_DIM(kernel, 0) + 1, PyArray_DIM(inputs, 2) - PyArray_DIM(kernel, 1) + 1, PyArray_DIM(kernel, 3)};
    PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNew(3, output_dims, NPY_DOUBLE);
    double* output_data = (double*) PyArray_DATA(output);
    double* inputs_data = (double*) PyArray_DATA(inputs);
    double* kernel_data = (double*) PyArray_DATA(kernel);

    for (int j = 0; j < output_dims[0]; ++j) {
        for (int k = 0; k < output_dims[1]; ++k) {
            for (int l = 0; l < output_dims[2]; ++l) {
                for (int m = 0; m < PyArray_DIM(kernel, 0); ++m) {
                    for (int n = 0; n < PyArray_DIM(kernel, 1); ++n) {
                        for (int o = 0; o < PyArray_DIM(inputs, 3); ++o) {
                            output_data[j * output_dims[1] * output_dims[2] + k * output_dims[2] + l] += inputs_data[i * PyArray_DIM(inputs, 1) * PyArray_DIM(inputs, 2) * PyArray_DIM(inputs, 3) + (j + m) * PyArray_DIM(inputs, 2) * PyArray_DIM(inputs, 3) + (k + n) * PyArray_DIM(inputs, 3) + o] * kernel_data[m * PyArray_DIM(kernel, 1) * PyArray_DIM(kernel, 2) * PyArray_DIM(kernel, 3) + n * PyArray_DIM(kernel, 2) * PyArray_DIM(kernel, 3) + o * PyArray_DIM(kernel, 3) + l];
                        }
                    }
                }
            }
        }
    }

    return (PyObject*) output;
}

static PyMethodDef methods[] = {
    {"conv2d_approx", (PyCFunction)conv2d_approx, METH_VARARGS, "Describe your function here"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "your_module",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_your_module(void) {
    import_array();
    return PyModule_Create(&module);
}
}
