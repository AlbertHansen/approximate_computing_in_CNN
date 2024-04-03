#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>

struct Metrics {
    double mse;
    double mae;
    double ed;
    double ep;
};

class Evaluator {
public:
    Evaluator(const std::vector<double>& expected, const std::vector<double>& actual);

    Metrics calculateMetrics() const;

private:
    std::vector<double> expected;
    std::vector<double> actual;

    double calculateMSE() const;
    double calculateMAE() const;
    double calculateED() const;
    double calculateEP(double epsilon) const;
};

#endif