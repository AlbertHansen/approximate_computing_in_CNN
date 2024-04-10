#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <iostream>
#include "Matrix.h"
#include "Perceptron.h"

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
    Sizes sizes;

public:
    //std::vector<Perceptron> filters; // Filters are represented as matrices
    ConvolutionalLayer(size_t inputSizeX, size_t inputSizeY, size_t numFilters, size_t filterSizeX, size_t filterSizeY);

    std::vector<Matrix> applyConvolution(const Matrix& input);
    void updateFilters(const std::vector<Matrix>& newFilters);
};

#endif
