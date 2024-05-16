#include "FullyConnectedLayer.h"
#include "FixedPointConverter.h"
#include <iostream>  // For debugging

FixedPointConverter<double> converter2(5, 50); // int type, 4 decimal bits, 4 fractional bits

FullyConnectedLayer::FullyConnectedLayer(size_t inputLength, size_t outputLength)
    : inputSize(inputLength), outputSize(outputLength) 
{
    // Initialize perceptrons in the layer with custom weights
    perceptrons.reserve(outputSize);  // Reserve space for outputSize perceptrons
}

void FullyConnectedLayer::setRelu(ActivationFunction<intmax_t> relu)
{
    this->relu = relu;
}

std::vector<intmax_t> FullyConnectedLayer::forward(const std::vector<intmax_t>& inputs, const std::vector<std::vector<intmax_t>>& weights, const std::vector<intmax_t>& biases) 
{
    perceptrons.clear();
    if (inputs.size() != inputSize /*|| biases.size() != outputSize*/) 
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

    std::vector<intmax_t> output;

    // Forward pass through the layer
    for (size_t i = 0; i < outputSize; i++) 
    {
        
        // Compute the output of each perceptron in the layer
        output.push_back(relu.ReLU(perceptrons.at(i).compute(biases.at(i))));
            
    }
    
    /*****************
    //std::vector<intmax_t> t = {perceptrons.at(1).compute(0)};
    std::cout << "size: " << perceptrons.at(0).getInputs().size() << std::endl;
    std::vector<intmax_t> test = /*converter2.convertToDouble(*perceptrons.at(0).getWeights()/*)*;
    std::cout << "sizetest: " << test.size() << std::endl;
    for (const auto& element : test)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    /*****************/

    return output;
}
