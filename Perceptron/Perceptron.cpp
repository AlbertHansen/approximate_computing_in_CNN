#include "Perceptron.h"

Perceptron::Perceptron() {
    // Default constructor, no additional initialization needed
}

Perceptron::Perceptron(const std::vector<intmax_t>& weights, const std::vector<intmax_t>& inputs) {
    this->weights = weights;
    this->inputs = inputs;
}

void Perceptron::setAdder(Adder adder) {
    this->adder = adder;
}

void Perceptron::setMultiplier(Multiplier multiplier) {
    this->multiplier = multiplier;
}

intmax_t Perceptron::compute(intmax_t bias) {
    intmax_t result = bias;
    for (uint16_t i = 0; i < weights.size(); ++i) {
        result = adder.add(result, multiplier.multiply(weights.at(i), inputs.at(i)));
        
    }
    return result;
}

void Perceptron::setWeights(const std::vector<intmax_t>& weights) {
    this->weights = weights;
}

void Perceptron::setInputs(const std::vector<intmax_t>& inputs) {
    this->inputs = inputs;
}
