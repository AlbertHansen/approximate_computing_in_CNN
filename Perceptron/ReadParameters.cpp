#include "ReadParameters.h"

ReadParameters::ReadParameters(const std::string& filenameWeights, const std::string& filenameBiases) {
    std::ifstream weightsFile(filenameWeights);
    std::ifstream biasFile(filenameBiases);
    if (!weightsFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filenameWeights << std::endl;
        
    }
    if (!biasFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filenameBiases << std::endl;
        
    }

    // Read each line from the file
    std::string line;
    while (getline(weightsFile, line)) {
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
        layer.weights.push_back(row);
    }
    weightsFile.close();
    while (getline(biasFile, line)) {
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
        layer.biases = row;
    }

    // Close the file
    biasFile.close();
}
