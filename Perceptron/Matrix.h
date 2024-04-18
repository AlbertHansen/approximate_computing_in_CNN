#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<int>& values);

    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    // Method to extract a sub-matrix
    Matrix extractSubMatrix(size_t startRow, size_t startCol, size_t subRows, size_t subCols) const;

    //Method to flatten the matrix
    std::vector<intmax_t> flatten() const;

};

#endif
