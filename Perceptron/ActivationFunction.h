#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <iostream>

template <typename T> 
class ActivationFunction {
public:
    // Bitwise ReLU
    intmax_t ReLU(T u) const {  // Add 'const' qualifier to the method
        const int lengthOfInput = sizeof(T) * 8;
        if (((u >> lengthOfInput-1)&1) == 1)
        {
            u=0;
        }
        
        return u;
    }

    intmax_t sigmoid(intmax_t ) const {
        return 1 / (1 + exp(-u));
    }
};

#endif
