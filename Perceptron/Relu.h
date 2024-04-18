#ifndef RELU_H
#define RELU_H

#include <iostream>

template <typename T> 
class Relu {
public:
    // Bitwise ReLU
    intmax_t ReLU(T u) {
        const int lengthOfInput = sizeof(T) * 8;
        if (((u >> lengthOfInput-1)&1) == 1)
        {
            u=0;
        }
        
        return u;
    }
};

#endif
