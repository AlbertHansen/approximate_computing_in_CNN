//ConvolutionalLayer.cpp
#include "ConvolutionalLayer.h"

// Constructor
ConvolutionalLayer::ConvolutionalLayer(size_t inputSizeX, size_t inputSizeY, size_t numFilters, size_t filterSizeX, size_t filterSizeY)
    : sizes{inputSizeX, inputSizeY, numFilters, filterSizeX, filterSizeY} {
    // Initialize filters with default values or specific initialization logic
    filters.resize(sizes.numFilters);
}



// Apply convolution using the initialized filters
std::vector<Matrix> ConvolutionalLayer::applyConvolution(const Matrix& input) 
{
    if (input.numCols() != sizes.inputSizeX || input.numRows() != sizes.inputSizeY) 
    {
        throw std::invalid_argument("Input size does not match initialised size.");
    }
    Perceptron perceptron(filters.at(0),biases);
    std::vector<Matrix> output;
    for (size_t k = 0; k < sizes.numFilters; k++)
    {
        Matrix featureMap(sizes.inputSizeX - sizes.filterSizeX + 1, sizes.inputSizeY - sizes.filterSizeY + 1);
        perceptron.setWeights(filters.at(k));
        for (size_t i = 0 ; i < featureMap.numCols(); i++)
        {
            for (size_t j = 0 ; j < featureMap.numRows(); j++)
            {
                //Matrix inputSubMatrix = input.extractSubMatrix(i,j,sizes.filterSizeX,sizes.filterSizeY);
                //std::vector<intmax_t> perceptronInput = inputSubMatrix.flatten();
                std::vector<intmax_t> perceptronInput = input.extractSubMatrix(i,j,sizes.filterSizeX,sizes.filterSizeY).flatten();
                
                perceptron.setInputs(perceptronInput);
                featureMap(i,j) = /*relu.ReLU(*/perceptron.compute(biases.at(k))/*)*/;
                
            }
            
        }
        output.push_back(featureMap);
    }
    return output;
}

// Update the filters with new values
void ConvolutionalLayer::updateFilters(const std::vector<Matrix>& newFilters, const std::vector<intmax_t> newBiases) 
{
    if (newFilters.size() != sizes.numFilters) 
    {
        std::cout << newFilters.size() << std::endl;
        throw std::invalid_argument("Number of new filters does not match the current number of filters.");
    }
    for (size_t k = 0; k < newFilters.size(); k++)
    {   
        std::vector<intmax_t> weights = newFilters.at(k).flatten();
        filters.at(k)= weights;
    }

    biases = newBiases;
}

std::vector<std::vector<intmax_t>> ConvolutionalLayer::getFilters() const {
    return filters;
}

std::vector<intmax_t> ConvolutionalLayer::getBiases() const {
    return biases;
}
