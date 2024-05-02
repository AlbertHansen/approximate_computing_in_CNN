#include <iostream>
#include <cstdint>
#include <vector>
#include "Evaluator.h"

typedef uint64_t (*BinaryOperation)(const uint64_t, const uint64_t);
/*********** Approx Adder ***************/
uint64_t add8se_8R9(const uint64_t B, const uint64_t A);

/*********** Accurate Adder ***********/
uint64_t add(const uint64_t B, const uint64_t A) {
    uint64_t result = B+A;
    return result;
}

std::vector<intmax_t> testAllCombinations(BinaryOperation operation) {
    std::vector<intmax_t> results;

    for (uint16_t A = 0; A <= 0xFF; ++A) {
        for (uint16_t B = 0; B <= 0xFF; ++B) {
            uint64_t result = operation(static_cast<uint64_t>(B), static_cast<uint64_t>(A));
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
    std::vector<intmax_t> Actual = testAllCombinations(add8se_8R9);

    writeVectorToCSV("Expected.csv",Expected);
    writeVectorToCSV("Actual.csv",Actual);

    Evaluator eval_add8se_8R9(Expected,Actual);
    Metrics add8se_8R9_metrics = eval_add8se_8R9.calculateMetrics();
    eval_add8se_8R9.writeMetricsToCSV("metrics.csv",add8se_8R9_metrics);    //(filename, evaluator.metrics)
    

    // Display the results (optional)
    for (const auto& row : Expected) {
        std::cout << row << " " << std::endl;
    }

    return 0;
}
