// FixedPointConverter.cpp
#include "FixedPointConverter.h"
#include <cmath>

template<typename T>
FixedPointConverter<T>::FixedPointConverter(int decimalBits, int fractionalBits)
    : decimalBits(decimalBits), fractionalBits(fractionalBits) {}

template<typename T>
std::vector<T> FixedPointConverter<T>::convertToFixedPoint(const std::vector<float>& input) const {
    std::vector<T> fixedPointValues;
    fixedPointValues.reserve(input.size());

    float scale = std::pow(2.0f, fractionalBits);
    for (float value : input) {
        fixedPointValues.push_back(static_cast<T>(value * scale));
    }

    return fixedPointValues;
}

// Explicit template instantiation for int type
template class FixedPointConverter<int>;
