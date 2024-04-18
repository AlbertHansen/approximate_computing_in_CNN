// FixedPointConverter.h
#ifndef FIXED_POINT_CONVERTER_H
#define FIXED_POINT_CONVERTER_H

#include <vector>

template<typename T>
class FixedPointConverter {
private:
    int decimalBits;
    int fractionalBits;

public:
    FixedPointConverter(int decimalBits, int fractionalBits);
    std::vector<T> convertToFixedPoint(const std::vector<float>& input) const;
};

#endif // FIXED_POINT_CONVERTER_H
