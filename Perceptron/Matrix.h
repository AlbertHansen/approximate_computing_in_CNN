#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include "Relu.h"

class Matrix {
private:
    std::vector<std::vector<intmax_t>> data;
    size_t rows;
    size_t cols;

    Relu<intmax_t> relu;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<intmax_t>& values);

    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    intmax_t& operator()(size_t i, size_t j);
    const intmax_t& operator()(size_t i, size_t j) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator+(intmax_t scalar) const;
    Matrix operator*(const Matrix& other) const;

    // Method to extract a sub-matrix
    Matrix extractSubMatrix(size_t startRow, size_t startCol, size_t subRows, size_t subCols) const;

    //Method to flatten the matrix
    std::vector<intmax_t> flatten() const;

    void unflatten(const std::vector<intmax_t>& flattened);

    Matrix applyRelu() const;

    
};

#endif
