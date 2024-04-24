#include "FullyConnectedLayer.h"
#include "Relu.h"
#include <iostream>  // For debugging

FullyConnectedLayer::FullyConnectedLayer(size_t inputLength, size_t outputLength)
    : inputSize(inputLength), outputSize(outputLength) 
{
    // Initialize perceptrons in the layer with custom weights
    perceptrons.reserve(outputSize);  // Reserve space for outputSize perceptrons
}

void FullyConnectedLayer::setRelu(Relu<intmax_t> relu)
{
    this->relu = relu;
}

std::vector<intmax_t> FullyConnectedLayer::forward(const std::vector<intmax_t>& inputs, const std::vector<std::vector<intmax_t>>& weights, const std::vector<intmax_t>& biases) 
{
    if (/*inputs.size() != inputSize ||*/ biases.size() != outputSize) 
    {
        std::cerr << "Input size mismatch in FullyConnectedLayer::forward()." << std::endl;
        return {};
    }
    for (size_t i = 0; i < outputSize; ++i) 
    {
        // Each perceptron has weights for each input feature plus a bias
        Perceptron p(weights.at(i),inputs);  // +1 for the bias term
        perceptrons.push_back(p);
    }

    std::vector<intmax_t> output(outputSize);

    // Forward pass through the layer
    for (size_t i = 0; i < outputSize; ++i) 
    {
        // Compute the output of each perceptron in the layer
        output.at(i) = relu.ReLU(perceptrons.at(i).compute(biases.at(i)));
    }

    return output;
}
