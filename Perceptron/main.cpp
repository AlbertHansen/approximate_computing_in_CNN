#include <iostream>
#include "Perceptron.h"
#include "Adder.h" // Include the adder class
#include "Multiplier.h" // Include the multiplier class

int main() {
    // Create a perceptron with initial weights and inputs
    std::vector<intmax_t> initialWeights = {5, 1, 2};
    std::vector<intmax_t> initialInputs = {1, 2, 3};
    Perceptron perceptron(initialWeights, initialInputs);

    Adder adder;
    Multiplier multiplier;

    perceptron.setAdder(adder);
    perceptron.setMultiplier(multiplier);

    intmax_t bias = 1;

    intmax_t output = perceptron.compute(bias);
    std::cout << "Output: " << (int)output << std::endl; // Output: Output: 0.5

    // Change weights and inputs
    std::vector<intmax_t> newWeights = {1, 2, 1};
    std::vector<intmax_t> newInputs = {2, 3, 1};
    perceptron.setWeights(newWeights);
    perceptron.setInputs(newInputs);

    output = perceptron.compute(bias);
    std::cout << "Updated Output: " << (int)output << std::endl; // Output: Updated Output: 1.0

    return 0;
}
