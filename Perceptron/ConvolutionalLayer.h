#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <iostream>
#include "Matrix.h"
#include "Perceptron.h"
#include "Relu.h"

class ConvolutionalLayer {
private:
    struct Sizes {
        size_t inputSizeX;
        size_t inputSizeY;
        size_t numFilters;
        size_t filterSizeX;
        size_t filterSizeY;
    };

    std::vector<std::vector<intmax_t>> filters; // Filters are represented as matrices
    std::vector<intmax_t> biases;
    Sizes sizes;

public:
    //std::vector<Perceptron> filters; // Filters are represented as matrices
    ConvolutionalLayer(size_t inputSizeX, size_t inputSizeY, size_t numFilters, size_t filterSizeX, size_t filterSizeY);

    std::vector<Matrix> applyConvolution(const Matrix& input);
    void updateFilters(const std::vector<Matrix>& newFilters, const std::vector<intmax_t> newBiases);

    std::vector<std::vector<intmax_t>> getFilters() const;
    std::vector<intmax_t> getBiases() const;
};

#endif
