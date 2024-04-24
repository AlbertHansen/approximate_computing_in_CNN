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

    Relu<intmax_t> relu;

    std::vector<std::vector<intmax_t>> filters; // Filters are represented as matrices
    std::vector<intmax_t> biases;
    Sizes sizes;

public:
    //std::vector<Perceptron> filters; // Filters are represented as matrices
    ConvolutionalLayer(size_t inputSizeX, size_t inputSizeY, size_t numFilters, size_t filterSizeX, size_t filterSizeY);

    void setRelu(Relu<intmax_t> relu);

    std::vector<Matrix> applyConvolution(const Matrix& input);
    void updateFilters(const std::vector<Matrix>& newFilters, const std::vector<intmax_t> newBiases);
};

#endif
