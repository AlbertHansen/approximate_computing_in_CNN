#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <iostream>
#include "Adder.h"
#include "Multiplier.h"

class Perceptron {
private:
    Adder adder;
    Multiplier multiplier;
    std::vector<intmax_t> weights;
    std::vector<intmax_t> inputs;

public:
    Perceptron();
    Perceptron(const std::vector<intmax_t>& weights, const std::vector<intmax_t>& inputs);
    
    void setAdder(Adder adder);
    void setMultiplier(Multiplier multiplier);

    intmax_t compute(intmax_t bias);

    const std::vector<intmax_t>& getWeights() const { return weights; }
    const std::vector<intmax_t>& getInputs() const { return inputs; }
    void setWeights(const std::vector<intmax_t>& weights);
    void setInputs(const std::vector<intmax_t>& inputs);
};

#endif
