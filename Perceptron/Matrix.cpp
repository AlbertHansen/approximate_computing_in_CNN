#include "Matrix.h"
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<intmax_t>(cols, 0.0));
    
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<intmax_t>& values) : rows(rows), cols(cols) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("Values size does not match matrix dimensions.");
    }

    data.resize(rows, std::vector<intmax_t>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = values[i * cols + j];
        }
    }
}

intmax_t& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Index out of range.");
    }
    return data[i][j];
}

const intmax_t& Matrix::operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Index out of range.");
    }
    return data[i][j];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions don't match.");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator+(intmax_t scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + scalar;
        }
    }
    return result;
}


Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication.");
    }

    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                result(i, j) += data[i][k] * other(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::extractSubMatrix(size_t startRow, size_t startCol, size_t subRows, size_t subCols) const {
    // Check if the start indices and sub-matrix size are within bounds
    if (startRow + subRows > rows || startCol + subCols > cols) {
        throw std::out_of_range("Sub-matrix indices are out of range.");
    }

    // Create a new matrix for the sub-matrix
    Matrix subMatrix(subRows, subCols);

    // Copy the elements from the original matrix to the sub-matrix
    for (size_t i = 0; i < subRows; ++i) {
        for (size_t j = 0; j < subCols; ++j) {
            subMatrix(i, j) = (*this)(startRow + i, startCol + j);
        }
    }

    return subMatrix;
}

std::vector<intmax_t> Matrix::flatten() const {
    std::vector<intmax_t> flattened;
    for (const auto& row : data) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

Matrix Matrix::applyRelu() const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Apply ReLU to each element
            result(i, j) = relu.ReLU((*this)(i, j)); // Assuming 'relu' is an instance of Relu<intmax_t>
        }
    }
    return result;
}

void Matrix::unflatten(const std::vector<intmax_t>& flattened) {
    if (flattened.size() != rows * cols) {
        throw std::invalid_argument("Flattened vector size does not match matrix dimensions.");
    }

    // Clear the existing data
    data.clear();
    data.resize(rows, std::vector<intmax_t>(cols));

    // Copy the elements from the flattened vector to the matrix
    size_t index = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = flattened[index++];
        }
    }
}
