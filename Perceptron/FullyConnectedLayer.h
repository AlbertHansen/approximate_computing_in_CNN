#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "Perceptron.h"
#include "Relu.h"
#include <vector>

class FullyConnectedLayer {
private:
    size_t inputSize;        // Size of input to the layer
    size_t outputSize;       // Size of output from the layer
    std::vector<Perceptron> perceptrons;  // Neurons in the layer
    Relu<intmax_t> relu;

public:
    FullyConnectedLayer(size_t inputLength, size_t outputLength);

    // Forward pass through the layer
    std::vector<intmax_t> forward(const std::vector<intmax_t>& inputs, const std::vector<std::vector<intmax_t>>& weights, const std::vector<intmax_t>& biases);

    void setRelu(Relu<intmax_t> relu);

    // Get the number of neurons in the layer
    size_t getNumNeurons() const { return perceptrons.size(); }
};

#endif // FULLY_CONNECTED_LAYER_H
