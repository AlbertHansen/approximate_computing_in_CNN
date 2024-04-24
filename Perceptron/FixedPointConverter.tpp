// FixedPointConverter.tpp
#ifndef FIXED_POINT_CONVERTER_TPP
#define FIXED_POINT_CONVERTER_TPP

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

template<typename T>
std::vector<T> FixedPointConverter<T>::convertToDouble(const std::vector<intmax_t>& input) const {
    std::vector<T> doubleValues;
    doubleValues.reserve(input.size());

    float scale = 1.0f / std::pow(2.0f, fractionalBits);
    for (T value : input) {
        doubleValues.push_back(static_cast<T>(value) * scale);
    }

    return doubleValues;
}

#endif // FIXED_POINT_CONVERTER_TPP
