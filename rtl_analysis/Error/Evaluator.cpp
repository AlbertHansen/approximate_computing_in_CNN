// Evaluator.cpp
#include "Evaluator.h"

Evaluator::Evaluator(const std::vector<intmax_t>& expected, const std::vector<intmax_t>& actual)
    : expected(expected), actual(actual) 
    {
        
    }

Metrics Evaluator::calculateMetrics() const {
    Metrics metrics;
    metrics.diff = calculateDIFF();
    metrics.mse = calculateMSE();
    metrics.mae = calculateMAE();
    metrics.wcd = calculateWCD(); 
    metrics.er = calculateER(); 
    metrics.hd = calculateBitwiseHDVector();
    metrics.mhd = calculateMHD();  
    return metrics;
}

std::vector<intmax_t> Evaluator::calculateDIFF() const
{
    std::vector<intmax_t> result;
    for (size_t i = 0; i < expected.size(); ++i) {
        result.push_back(expected.at(i) - actual.at(i));
    }
    return result;
}

double Evaluator::calculateMSE() const 
{
    intmax_t sum = 0;  // Change sum to integer type
    for (size_t i = 0; i < expected.size(); ++i) {
        intmax_t diff = expected.at(i) - actual.at(i);  // Change diff to integer type
        sum += diff * diff;
    }

    return static_cast<double>(sum) / expected.size();  // Convert sum back to double for division
}


double Evaluator::calculateMAE() const {
    intmax_t sum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        sum += std::abs(expected.at(i) - actual.at(i));
    }
    return static_cast<double>(sum) / expected.size();
}

double Evaluator::calculateWCD() const {
    intmax_t maxDiff = std::numeric_limits<intmax_t>::min(); // Initialize maxDiff to smallest possible value

    // Calculate absolute differences and find maximum
    for (size_t i = 0; i < expected.size(); ++i) 
    {
        intmax_t diff = std::abs(expected[i] - actual[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

double Evaluator::calculateER() const {
    size_t incorrectCount = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected.at(i) != actual.at(i)) {
            incorrectCount++;
        }
    }
    return (static_cast<double>(incorrectCount) / expected.size()) * 100.0;
}

std::vector<size_t> Evaluator::calculateBitwiseHDVector() const {
    std::vector<size_t> hammingDistances;
    for (size_t i = 0; i < expected.size(); ++i) {
        // Get the bitwise XOR of expected[i] and actual[i]
        intmax_t xorResult = expected[i] ^ actual[i];
        size_t hammingDistance = 0;
        // Count the set bits in xorResult
        while (xorResult > 0) {
            hammingDistance += xorResult & 1; // Add 1 if the rightmost bit is 1
            xorResult >>= 1; // Shift right to check the next bit
        }
        hammingDistances.push_back(hammingDistance);
    }
    return hammingDistances;
}

double Evaluator::calculateMHD() const {
    std::vector<size_t> hammingDistances = calculateBitwiseHDVector();
    
    size_t sum = 0;
    for (size_t hd : hammingDistances) {
        sum += hd;
    }
    
    return static_cast<double>(sum) / hammingDistances.size();
}

void Evaluator::writeMetricsToCSV(const std::string& filename, const Metrics& metrics) const {
        std::ofstream outfile(filename);
        if (!outfile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Write header
        outfile << "mse,mae,wcd,er,mhd" << std::endl;

        // Write data
        outfile << metrics.mse << ","
                << metrics.mae << ","
                << metrics.wcd << ","
                << metrics.er << ","
                << metrics.mhd << std::endl;

        outfile.close();
        //std::cout << "Metrics written to " << filename << std::endl;
    }
