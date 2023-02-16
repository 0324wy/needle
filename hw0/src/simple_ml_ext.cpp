#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

namespace py = pybind11;

float *matrix_slice_by_raw(const float *X, size_t m, size_t n, size_t start_index, size_t batch) {
    size_t slice_nums = min(m - start_index * batch, batch);

    float *result = new float [slice_nums * n];
    for (int i = 0; i < slice_nums * n; ++i) {
        result[i] = X[i + start_index * batch * n];
    }
    return result;
}

unsigned char *vector_slice_by_raw(const unsigned char *y, size_t m, size_t start_index, size_t batch) {

    size_t slice_nums = min(m - start_index * batch, batch);

    unsigned char *result = new unsigned char[slice_nums];
    for (int i = 0; i < slice_nums; ++i) {
        result[i] = y[i + start_index * batch];
    }
    return result;
}


float *dot_product(const float *X, float *theta, size_t m, size_t n, size_t k) {
    // 1. x * theta : (m * k) = > (k * i + j)
    // x: (m * n) => (n * i + j)
    // theta: (n * k) => (k * i + j)

    float *x_theta = new float [m * k];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            x_theta[i * k + j] = 0;
            for (int l = 0; l < n; ++l) {
                x_theta[i * k + j] = X[n * i + l] * theta[k * l + j] + x_theta[i * k + j];
            }
        }
    }
    return x_theta;
}

float *softmax(const float *x_theta, size_t m, size_t k) {
    float *x_exp = new float [m * k];
    for (int i = 0; i < m; ++i) {
        float sum_i = 0;
        for (int j = 0; j < k; ++j) {
            sum_i += exp(x_theta[k * i + j]);
        }

        for (int j = 0; j < k; ++j) {
            x_exp[k * i + j] = exp(x_theta[k * i + j]) / sum_i;
        }
    }
    return x_exp;
}

float *one_hot(const unsigned char *y, size_t m, size_t k) {
    float *oneHotResult = new float [m * k];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            oneHotResult[i * k + j] = 0;
        }
        oneHotResult[i * k + y[i]] = 1;
    }
    return oneHotResult;
}

float *matrix_minus(const float *x_softmax, const float *oneHotResult, size_t m, size_t k) {
    float *result = new float [m * k];
    for (int i = 0; i < m * k; ++i) {
        result[i] = x_softmax[i] - oneHotResult[i];
    }
    return result;
}

float *matrix_minus_in_place(float *theta, const float *grid, size_t m, size_t k) {

    for (int i = 0; i < m * k; ++i) {
        theta[i] = theta[i] - grid[i];
    }
    return theta;

}

float *matrix_transpose(const float *x, size_t m, size_t n) {
    float *result = new float [m * n];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j * m + i] = x[i * n + j];
        }
    }
    return result;
}


float *matrix_multiply_scalar(const float *x, size_t m, size_t k, float number) {
    float *result = new float [m * k];
    for (int i = 0; i < m * k; ++i) {
        result[i] = x[i] * number;
    }
    return result;
}

int get_batch_nums(size_t batch, size_t m) {
    int batch_nums;
    size_t b = m / batch; // batch
    if (m % batch == 0) {
        batch_nums = b;
    } else {
        batch_nums = b + 1;
    }
    return batch_nums;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    int batch_nums = get_batch_nums(batch, m);

    for (int i = 0; i < batch_nums; ++i) {

        size_t len_x_batch = min(m - i * batch, batch);
        float *x_batch = matrix_slice_by_raw(X, m, n, i, batch);

        unsigned char *y_batch = vector_slice_by_raw(y, m, i, batch);

        float *x_theta = dot_product(x_batch, theta, len_x_batch, n, k);

        float *x_softmax = softmax(x_theta, len_x_batch, k);

        float *y_one_hot = one_hot(y_batch, len_x_batch, k);

        float *x_minus = matrix_minus(x_softmax, y_one_hot, len_x_batch, k);

        float *x_transpose = matrix_transpose(x_batch, len_x_batch, n);

        float *grid = dot_product(x_transpose, x_minus, n, len_x_batch, k);

        float *grid_divided = matrix_multiply_scalar(grid, n, k, 1.0 / len_x_batch);

        theta = matrix_minus_in_place(theta, matrix_multiply_scalar(grid_divided, n, k, lr), n, k);

    }

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
