#include "Relu.h"

int main ()
{
    Relu<int> relu;
    int inputTest = 50;
    std::cout << relu.ReLU(inputTest) << std::endl;
    return 0;
}