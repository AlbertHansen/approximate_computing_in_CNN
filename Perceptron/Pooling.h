#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "Matrix.h"

class PoolingLayer {
private:
    size_t poolSizeX;  // Pooling window size along X axis
    size_t poolSizeY;  // Pooling window size along Y axis

public:
    PoolingLayer(size_t sizeX, size_t sizeY);

    std::vector<Matrix> applyMaxPooling(const std::vector<Matrix>& input);
};

#endif  // POOLING_LAYER_H
