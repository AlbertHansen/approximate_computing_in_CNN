#include "ConvolutionalLayer.h"

// Constructor
ConvolutionalLayer::ConvolutionalLayer(size_t inputSizeX, size_t inputSizeY, size_t numFilters, size_t filterSizeX, size_t filterSizeY)
    : sizes{inputSizeX, inputSizeY, numFilters, filterSizeX, filterSizeY} {
    // Initialize filters with default values or specific initialization logic
    filters.resize(sizes.filterSizeX*sizes.filterSizeY);
}

// Apply convolution using the initialized filters
std::vector<Matrix> ConvolutionalLayer::applyConvolution(const Matrix& input) 
{
    if (input.numCols() != sizes.inputSizeX || input.numRows() != sizes.inputSizeY) 
    {
        throw std::invalid_argument("Input size does not match initialised size.");
    }
    
    std::vector<Matrix> output;
    for (size_t k = 0; k < sizes.numFilters; k++)
    {
        Matrix featureMap(sizes.inputSizeX - sizes.filterSizeX + 1, sizes.inputSizeY - sizes.filterSizeY + 1);
        for (size_t i = 0 ; i < featureMap.numCols(); i++)
        {
            for (size_t j = 0 ; j < featureMap.numRows(); j++)
            {
                Matrix inputSubMatrix = input.extractSubMatrix(i,j,sizes.filterSizeX,sizes.filterSizeY);
                std::vector<intmax_t> perceptronInput = inputSubMatrix.flatten();
                Perceptron perceptron(filters.at(k),perceptronInput);
                featureMap(i,j) = perceptron.compute(0);
            }
            
        }
        output.push_back(featureMap);
    }
    return output;
}

// Update the filters with new values
void ConvolutionalLayer::updateFilters(const std::vector<Matrix>& newFilters) 
{
    if (newFilters.size() != sizes.numFilters) 
    {
        throw std::invalid_argument("Number of new filters does not match the current number of filters.");
    }
    for (size_t k = 0; k < newFilters.size(); k++)
    {        
        std::vector<intmax_t> weights = newFilters.at(k).flatten();
        filters.at(k)= weights;
        
    }
}
