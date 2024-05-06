#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> worker(int i, py::array_t<double> inputs, py::array_t<double> kernel) {
    auto inputs_info = inputs.request();
    auto kernel_info = kernel.request();

    if (inputs_info.ndim != 4 || kernel_info.ndim != 4)
        throw std::runtime_error("Number of dimensions must be four");

    // Create a new PyArray to store the result
    std::vector<ssize_t> shape = {inputs_info.shape[0] - kernel_info.shape[0] + 1,
                                  inputs_info.shape[1] - kernel_info.shape[1] + 1,
                                  kernel_info.shape[3]};
    py::array_t<double> result(shape);

    auto r = result.mutable_unchecked<3>();
    double *inputs_ptr = static_cast<double *>(inputs_info.ptr);
    double *kernel_ptr = static_cast<double *>(kernel_info.ptr);

    for (ssize_t j = 0; j < r.shape(0); ++j) {
        for (ssize_t k = 0; k < r.shape(1); ++k) {
            for (ssize_t l = 0; l < r.shape(2); ++l) {
                for (ssize_t m = 0; m < kernel_info.shape[0]; ++m) {
                    for (ssize_t n = 0; n < kernel_info.shape[1]; ++n) {
                        for (ssize_t o = 0; o < inputs_info.shape[3]; ++o) {
                            r(j, k, l) += inputs_ptr[i * inputs_info.strides[0] + (j+m) * inputs_info.strides[1] + (k+n) * inputs_info.strides[2] + o * inputs_info.strides[3]] * 
                                          kernel_ptr[m * kernel_info.strides[0] + n * kernel_info.strides[1] + o * kernel_info.strides[2] + l * kernel_info.strides[3]];
                        }
                    }
                }
            }
        }
    }

    return result;
}

PYBIND11_MODULE(test, m) {
    m.def("worker", &worker, "A function which computes a 3D array from an integer and two 4D arrays");
}
