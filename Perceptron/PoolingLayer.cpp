#include "PoolingLayer.h"
#include "Matrix.h"  // Assuming you have a Matrix class defined
#include <algorithm>  // For std::max

PoolingLayer::PoolingLayer(size_t sizeX, size_t sizeY) : poolSizeX(sizeX), poolSizeY(sizeY) {}

std::vector<Matrix> PoolingLayer::applyMaxPooling(const std::vector<Matrix>& input) 
{
    size_t inputSizeX = input.at(0).numRows();
    size_t inputSizeY = input.at(0).numCols();

    std::vector<Matrix> pooledOutput(inputSizeX / poolSizeX, inputSizeY / poolSizeY);

    // Apply max pooling
    for (size_t k = 0; k < input.size(); k++)
    {
        for (size_t i = 0; i < inputSizeX; i += poolSizeX) 
        {
            for (size_t j = 0; j < inputSizeY; j += poolSizeY) 
            {
                double maxVal = input.at(k)(i, j);  // Initialize maxVal with top-left element of window

                // Find maximum value within the pooling window
                for (size_t x = i; x < i + poolSizeX; ++x) 
                {
                    for (size_t y = j; y < j + poolSizeY; ++y) 
                    {
                        maxVal = std::max(maxVal, input.at(k)(x, y));
                    }
                }

            // Assign the maximum value to the corresponding position in the output
            pooledOutput(i / poolSizeX, j / poolSizeY) = maxVal;
            }
        }
    }
    

    return pooledOutput;
}
