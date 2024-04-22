#include "FullyConnectedLayer.h"
#include <iostream>  // For debugging

FullyConnectedLayer::FullyConnectedLayer(size_t inputSize, size_t outputSize, const std::vector<std::vector<double>>& weights)
    : inputSize(inputSize), outputSize(outputSize) 
{
    // Initialize perceptrons in the layer with custom weights
    perceptrons.reserve(outputSize);  // Reserve space for outputSize perceptrons
    for (size_t i = 0; i < outputSize; ++i) {
        // Each perceptron has weights for each input feature plus a bias
        Perceptron p(inputSize + 1, weights[i]);  // +1 for the bias term
        perceptrons.push_back(p);
    }
}

std::vector<double> FullyConnectedLayer::forward(const std::vector<double>& input) {
    if (input.size() != inputSize) {
        std::cerr << "Input size mismatch in FullyConnectedLayer::forward()." << std::endl;
        return {};
    }

    std::vector<double> output(outputSize);

    // Forward pass through the layer
    for (size_t i = 0; i < outputSize; ++i) {
        // Compute the output of each perceptron in the layer
        output[i] = perceptrons[i].compute(input);
    }

    return output;
}
