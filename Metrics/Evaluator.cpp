// Evaluator.cpp
#include "Evaluator.h"

Evaluator::Evaluator(const std::vector<double>& expected, const std::vector<double>& actual)
    : expected(expected), actual(actual) 
    {

    }

Metrics Evaluator::calculateMetrics() const {
    Metrics metrics;
    metrics.mse = calculateMSE();
    metrics.mae = calculateMAE();
    metrics.ed = calculateED();
    metrics.ep = calculateEP(0.5); // Assuming epsilon is 0.5
    return metrics;
}

double Evaluator::calculateMSE() const {
    double sum = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double diff = expected[i] - actual[i];
        sum += diff * diff;
    }
    return sum / expected.size();
}

double Evaluator::calculateMAE() const {
    double sum = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        sum += std::abs(expected[i] - actual[i]);
    }
    return sum / expected.size();
}

double Evaluator::calculateED() const {
    double sum = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double diff = expected[i] - actual[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double Evaluator::calculateEP(double epsilon) const {
    size_t count = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) <= epsilon) {
            ++count;
        }
    }
    return static_cast<double>(count) / expected.size() * 100.0;
}
