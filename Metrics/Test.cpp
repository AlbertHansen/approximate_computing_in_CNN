#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>

void writeMatrixToCSV(const std::string& filename, const std::vector<std::vector<int>>& matrix) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix.at(i).size(); ++j) {
            outputFile << matrix[i][j];
            if (j != matrix[i].size() - 1) {
                outputFile << ","; // Add comma except for the last element in a row
            }
        }
        outputFile << std::endl; // Move to the next line after each row
    }

    outputFile.close();
    //std::cout << "Matrix written to " << filename << " successfully." << std::endl;
}

int main()
{
    std::vector<std::vector<int>> results;
    for (int i = -128; i < 128 ; i++)
    {
        std::vector<int> result;
        for (int j = -128; j < 128; j++)
        {
            result.push_back(j+i);
        }
        results.push_back(result);
    }
    writeMatrixToCSV("LUT.csv",results);
};