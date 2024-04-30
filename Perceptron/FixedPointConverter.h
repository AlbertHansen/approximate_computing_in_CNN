// FixedPointConverter.h
#ifndef FIXED_POINT_CONVERTER_H
#define FIXED_POINT_CONVERTER_H

#include <vector>
#include <iostream>

template<typename T>
class FixedPointConverter {
private:
    int decimalBits;
    int fractionalBits;

public:
    FixedPointConverter(int decimalBits, int fractionalBits);
    std::vector<T> convertToFixedPoint(const std::vector<float>& input) const;
    std::vector<T> convertToDouble(const std::vector<intmax_t>& input) const;
    std::vector<T> truncateLSBs(const std::vector<intmax_t>& input, int fractionalOut);
};

// Include template implementation here
#include "FixedPointConverter.tpp"

#endif // FIXED_POINT_CONVERTER_H
