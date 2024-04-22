#include "PoolingLayer.h"
#include "Matrix.h"  // Assuming you have a Matrix class defined
#include <algorithm>  // For std::max

PoolingLayer::PoolingLayer(size_t sizeX, size_t sizeY) : poolSizeX(sizeX), poolSizeY(sizeY) {}

std::vector<Matrix> PoolingLayer::applyMaxPooling(const std::vector<Matrix>& input) 
{
    size_t inputSizeX = input.at(0).numRows();
    size_t inputSizeY = input.at(0).numCols();

    std::vector<Matrix> pooledOutput;

    // Apply max pooling
    for (size_t k = 0; k < input.size(); k++)
    {
        size_t outputSizeX = inputSizeX / poolSizeX;
        size_t outputSizeY = inputSizeY / poolSizeY;
        Matrix pooledOutputIntermediate(outputSizeX, outputSizeY);

        for (size_t i = 0; i < outputSizeX; i++) 
        {
            for (size_t j = 0; j < outputSizeY; j++) 
            {
                size_t startX = i * poolSizeX;
                size_t endX = std::min(startX + poolSizeX, inputSizeX);
                size_t startY = j * poolSizeY;
                size_t endY = std::min(startY + poolSizeY, inputSizeY);

                double maxVal = input.at(k)(startX, startY);  // Initialize maxVal with top-left element of window

                // Find maximum value within the pooling window
                for (size_t x = startX; x < endX; ++x) 
                {
                    for (size_t y = startY; y < endY; ++y) 
                    {
                        maxVal = std::max(maxVal, input.at(k)(x, y));
                    }
                }

                // Assign the maximum value to the corresponding position in the output
                pooledOutputIntermediate(i, j) = maxVal;
            }
        }
        pooledOutput.push_back(pooledOutputIntermediate);
    }

    return pooledOutput;
}

