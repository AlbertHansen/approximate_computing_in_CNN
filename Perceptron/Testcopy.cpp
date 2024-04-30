#include <iostream>
#include "Perceptron.h"
#include "ConvolutionalLayer.h" 
#include "Matrix.h"
#include "Adder.h"
#include <vector>
#include "FixedPointConverter.h"
#include "FullyConnectedLayer.h"
#include "PoolingLayer.h"
#include "ReadParameters.h"
#include <fstream>
#include <sstream>
#include <string>

template <typename T>
void printBits(T value) 
{
    const int totalBits = sizeof(T) * 8;
    for (int i = totalBits - 1; i >= 0; --i) 
    {
        std::cout << ((value >> i) & 1);
    }
    std::cout << std::endl;
}

class PartitionUtility {
public:
    // Method to partition a vector into subvectors of n elements each
    template<typename T>
    static std::vector<std::vector<T>> partitionVector(const std::vector<T>& inputVector, size_t n) {
        std::vector<std::vector<T>> result;
        size_t size = inputVector.size();
        size_t numSubVectors = (size + n - 1) / n;  // Ceiling division to calculate number of subvectors

        result.reserve(numSubVectors);

        for (size_t i = 0; i < size; i += n) {
            auto subBegin = inputVector.begin() + i;
            auto subEnd = inputVector.begin() + std::min(i + n, size);
            result.emplace_back(subBegin, subEnd);
        }

        return result;
    }
};

std::vector<std::vector<float>> readInput(const std::string& filename) {
    std::vector<std::vector<float>> data;

    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;  // Return empty data
    }

    // Read each line from the file
    std::string line;
    while (getline(file, line)) {
        // Create a stringstream from the line
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        // Read each cell from the stringstream
        while (getline(ss, cell, ',')) {
            // Convert cell to float
            row.push_back(stof(cell));
        }

        // Add the row to the data vector
        data.push_back(row);
    }

    // Close the file
    file.close();

    return data;
}

int main()
{
    std::vector<std::vector<float>> inputBatch = readInput("");

    FullyConnectedLayer dense(4,1);
    std::vector<intmax_t> testInp = {1,10,-100,1000};
    std::vector<intmax_t> testWeight = {3,6,9,12};
    std::vector<intmax_t> testBias(4,0);
    std::vector<std::vector<intmax_t>> tW;
    tW.push_back(testWeight);

    std::cout << "hej: " << dense.forward(testInp,tW,testBias).at(0);
}
