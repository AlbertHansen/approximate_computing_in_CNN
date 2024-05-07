#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

struct Metrics {
    std::vector<intmax_t> diff;
    double mse;         //Mean square error
    double mae;         //Mean Absolute error
    double wcd;         //Worst-case error
    double mre;        //Average relative error magnitude
    double er;          //ErrorRate
    double mhd;         //Mean Hamming distance
    std::vector<size_t> hd;
};

class Evaluator {
public:
    Evaluator(const std::vector<intmax_t>& expected, const std::vector<intmax_t>& actual);

    Metrics calculateMetrics() const;
    void writeMetricsToCSV(const std::string& filename, const Metrics& metrics) const;

private:
    std::vector<intmax_t> expected;
    std::vector<intmax_t> actual;

    std::vector<intmax_t> calculateDIFF() const;
    double calculateMSE() const;
    double calculateMAE() const;
    double calculateWCD() const;
    double calculateMRE() const;
    double calculateER() const;
    std::vector<size_t> calculateBitwiseHDVector() const;
    double calculateMHD() const;
};

#endif