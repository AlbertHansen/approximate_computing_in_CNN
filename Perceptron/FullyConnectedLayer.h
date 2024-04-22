#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "Perceptron.h"
#include <vector>

class FullyConnectedLayer {
private:
    size_t inputSize;        // Size of input to the layer
    size_t outputSize;       // Size of output from the layer
    std::vector<Perceptron> perceptrons;  // Neurons in the layer

public:
    FullyConnectedLayer(size_t inputSize, size_t outputSize, const std::vector<std::vector<double>>& weights);

    // Forward pass through the layer
    std::vector<double> forward(const std::vector<double>& input);

    // Get the number of neurons in the layer
    size_t getNumNeurons() const { return perceptrons.size(); }
};

#endif // FULLY_CONNECTED_LAYER_H
