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
    if (inputs.size() != weights.size()) 
    {
        std::cerr << "Input and Weights in perceptron mismatch in size." << std::endl;
        return {};
    }
    intmax_t result = bias;
    for (uint16_t i = 0; i < weights.size(); ++i) {
        result = adder.add(result, multiplier.mul8s_1L12(weights.at(i), inputs.at(i)));
        //result = adder.add(result, static_cast<intmax_t>(multiplier.mul8s_1KV9(static_cast<int8_t>(weights.at(i)), static_cast<int8_t>(inputs.at(i)))));
        
        //std::cout << "W: " << weights.at(i) << " I: " << inputs.at(i) << " R: " << result << std::endl;
        
    }
    //std::cout << result << ",";
    //std::cout << std::endl;
    return result;
}

void Perceptron::setWeights(const std::vector<intmax_t>& weights) {
    this->weights = weights;
}

void Perceptron::setInputs(const std::vector<intmax_t>& inputs) {
    this->inputs = inputs;
}
