#ifndef READ_PARAMETERS_H
#define READ_PARAMETERS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

struct LayerParams 
{
        std::vector<std::vector<float>> weights;  // Weights matrix for the layer
        std::vector<float> biases;  // Biases vector for the layer
};

class ReadParameters 
{
private:
    LayerParams layer;  // Vector to store parameters for each layer

public:
    // Method to read weights and biases from a CSV file
    ReadParameters(const std::string& filenameWeights, const std::string& filenameBiases);

    // Getter for accessing layer parameters
    const LayerParams& getLayer() const { return layer; }
};

#endif
