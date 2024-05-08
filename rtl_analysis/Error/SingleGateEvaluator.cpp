#include <iostream>
#include <cstdint>
#include <vector>
#include <cstdint>
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
/*********** Approx multiplier ***************/
uint64_t add8se_839(const uint64_t B, const uint64_t A);

/*********** Accurate multiplier ***********/
uint64_t add(const uint64_t B, const uint64_t A) {
    uint64_t result = B+A;
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
    std::vector<intmax_t> results;

    for (int16_t signedA = -128; signedA <= -127; ++signedA) {
        for (int16_t signedB = -128; signedB <= -127; ++signedB) {
            uint64_t A = static_cast<uint64_t>(signedA);
            uint64_t B = static_cast<uint64_t>(signedB);
            std::cout << "A: " << A << " ";
            //printBits(A);
            std::cout << "B: " << B << " ";
            //printBits(B);

            uint64_t result = operation(B, A); // Note the order of arguments
            //std::cout << "R: " << result << " ";
            //printBits(result);
            results.push_back(static_cast<intmax_t>(result));
        }
    }
    return results;
}

std::vector<intmax_t> testAllCombinationsAccurate(BinaryOperation operation) {
    std::vector<intmax_t> results;

    for (int16_t signedA = -128; signedA <= 127; ++signedA) {
        for (int16_t signedB = -128; signedB <= 127; ++signedB) {
            int8_t A = static_cast<int8_t>(signedA);
            int8_t B = static_cast<int8_t>(signedB);

            intmax_t result = static_cast<intmax_t>(operation(B, A)); // Note the order of arguments
            
            results.push_back(result);
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
}

int main() {
    
    std::vector<intmax_t> Expected = testAllCombinations(add);
    std::vector<intmax_t> Actual = testAllCombinations(add8se_839);
    
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
