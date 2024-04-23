#include "Relu.h"
#include "FullyConnectedLayer.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

class PartitionUtility {
public:
    // Method to partition a vector into subvectors of n elements each
    template<typename T>
    static std::vector<std::vector<T>> partitionVector(const std::vector<T>& inputVector, size_t n) {
        std::vector<std::vector<T>> result;
        size_t size = inputVector.size();
        size_t numSubVectors = (size + n - 1) / n;  // Ceiling division to calculate number of subvectors

        result.reserve(numSubVectors);

        for (size_t i = 0; i < size; i += n) {
            auto subBegin = inputVector.begin() + i;
            auto subEnd = inputVector.begin() + std::min(i + n, size);
            result.emplace_back(subBegin, subEnd);
        }

        return result;
    }
};

int main ()
{   
    std::vector<std::vector<intmax_t>> wee;
    std::vector<intmax_t> inp = {2,12,2,2,2};
    std::vector<intmax_t> WWW = {1,1,1,1,1};
    wee.push_back(WWW);
    std::vector<intmax_t> bi = {0};
    FullyConnectedLayer dense(5,1);

    std::vector<intmax_t> res = dense.forward(inp,wee,bi);

    std::cout << res.at(0) << std::endl;
}