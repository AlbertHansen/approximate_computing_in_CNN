#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cstdint>
#include <string>
#include <filesystem>
#include "Evaluator.h"

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

typedef uint64_t (*BinaryOperation)(const uint64_t, const uint64_t);
//typedef int16_t (*BinaryOperation)(const int8_t, const int8_t);

/*********** Approx adder ***************/
uint64_t add8se_8NH(const uint64_t B, const uint64_t A);

/*********** Accurate adder ***********/
uint64_t add(const uint64_t B, const uint64_t A) {
    uint64_t result = B+A;
    return result;  // Convert result back to uint64_t before returning
}
/*********** Approx multiplier ***************/
int16_t mul8s_1L2J(const int8_t B, const int8_t A);

/*********** Accurate multiplier ***********/
int16_t mult(const int8_t B, const int8_t A) {
    uint64_t result = B*A;
    return result;  // Convert result back to uint64_t before returning
}

// Sign extension function
intmax_t signExtend(uint64_t result, uint64_t signBit) {
    uint64_t signMask = 1ULL << (signBit - 1);
    return static_cast<intmax_t>((result ^ signMask) - signMask);
}

// Conversion function
std::vector<intmax_t> convertToSigned(const std::vector<uint64_t>& results, uint64_t signBit) {
    std::vector<intmax_t> signedResults;
    for (const auto& result : results) {
        signedResults.push_back(signExtend(result, signBit));
    }
    return signedResults;
}

std::vector<intmax_t> testAllCombinations(BinaryOperation operation) {
    //std::vector<intmax_t> results;
    std::vector<uint64_t> results;
    for (intmax_t signedA = -128; signedA <= 127; ++signedA) {
        for (intmax_t signedB = -128; signedB <= 127; ++signedB) {
            uint64_t A = static_cast<uint64_t>(signedA);
            uint64_t B = static_cast<uint64_t>(signedB);
            
            
            uint64_t result = operation(B, A); // Note the order of arguments
            results.push_back(result);
            //results.push_back(static_cast<intmax_t>(operation(signedB,signedA)));
        }
    
    }
    return convertToSigned(results,9);
}

std::vector<intmax_t> testAllCombinationsAccurate(BinaryOperation operation) {
    //std::vector<uint64_t> results;
    std::vector<intmax_t> results;
    for (intmax_t signedA = -128; signedA <= 127; ++signedA) {
        for (intmax_t signedB = -128; signedB <= 127; ++signedB) {
            uint64_t A = static_cast<uint64_t>(signedA);
            uint64_t B = static_cast<uint64_t>(signedB);
            

            uint64_t result = operation(B, A); // Note the order of arguments
            
            results.push_back((intmax_t)result);
        }
    }
    return results;
}

void writeVectorToCSV(const std::string& filename, const std::vector<intmax_t>& data) {
    std::ofstream outfile(filename);
    // Write data to CSV
    for (size_t i = 0; i < data.size(); ++i) {
        outfile << data[i];
        if (i != data.size() - 1) {
            outfile << ",";  // Add comma if not the last element
        }
    }
    outfile << std::endl;
    outfile.close();
    std::cout << "Data written to " << filename << std::endl;
}/*

void writeVectorToCSV(const std::string& filename, const std::vector<intmax_t>& data) {
    std::ofstream outfile;
    if (std::filesystem::exists(filename)) {
        // Append to existing file
        outfile.open(filename, std::ios_base::app);
        outfile << std::endl;  // Add new line if file already has data
    } else {
        // Create new file
        outfile.open(filename);
    }

    // Write data to CSV
    for (size_t i = 0; i < data.size(); ++i) {
        outfile << data[i];
        if (i != data.size() - 1) {
            outfile << ",";  // Add comma if not the last element
        }
    }
    outfile << std::endl;
    outfile.close();
    std::cout << "Data written to " << filename << std::endl;
}
*/
int main() {

    
        std::vector<intmax_t> Expected = testAllCombinationsAccurate(add);
        std::vector<intmax_t> Actual = testAllCombinations(add8se_8NH);
    
        writeVectorToCSV("./Error/Error_files/Expected.csv",Expected);
        writeVectorToCSV("./Error/Error_files/Actual.csv",Actual);

        Evaluator eval_add8se_8R9(Expected,Actual);
        Metrics mul8s_1KV9_metrics = eval_add8se_8R9.calculateMetrics();
        eval_add8se_8R9.writeMetricsToCSV("./Error/Error_files/metrics.csv",mul8s_1KV9_metrics);    //(filename, evaluator.metrics)
    

    /* Display the results (optional)
    for (const auto& row : Expected) {
        std::cout << row << " " << std::endl;
    }*/

    return 0;
}
