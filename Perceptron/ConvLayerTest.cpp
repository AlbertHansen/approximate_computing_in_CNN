#include <iostream>
#include "Perceptron.h"
#include "ConvolutionalLayer.h" 
#include "Matrix.h"
#include "Adder.h"
#include <vector>
#include "FixedPointConverter.h"
#include "Pooling.h"

template <typename T>
void printBits(T value) 
{
    const int totalBits = sizeof(T) * 8;
    for (int i = totalBits - 1; i >= 0; --i) 
    {
        std::cout << ((value >> i) & 1);
    }
    std::cout << std::endl;
}

int main()
{
    
    std::vector<float> flattenedInput =     {-3.5, -1.5, -1.0, 5.1, 2.1,
                                                1.0, 1.0, 1.0, 4.1, 1.1,
                                                1.0, 1.0, 1.0, 4.1, 5.1,
                                                5.1, 2.1, 1.1, 4.1, 4.1,
                                                1.1, 1.1, 2.1, 2.1, 3.1};
    
    std::vector<float> flattenedFilter1 =   {1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0};
    std::vector<float> flattenedFilter2 =   {1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0};
    

    FixedPointConverter<int> converter(4, 4); // int type, 0 decimal bits, 8 fractional bits

    
    std::vector<int> fixedInput = converter.convertToFixedPoint(flattenedInput);
    std::vector<int> fixedFilt1 = converter.convertToFixedPoint(flattenedFilter1);
    std::vector<int> fixedFilt2 = converter.convertToFixedPoint(flattenedFilter2);    
    
    
    Matrix inputMatrix(5,5,fixedInput);
    Matrix filterMatrix1(3,3,fixedFilt1);
    Matrix filterMatrix2(3,3,fixedFilt2);
    std::vector<Matrix> newfilters = {filterMatrix1,filterMatrix2};

    

    size_t inputX = inputMatrix.numCols();
    size_t inputY = inputMatrix.numRows();
    size_t numKernels = 2;
    size_t kernelX = filterMatrix1.numCols();
    size_t kernelY = filterMatrix1.numRows();

    ConvolutionalLayer layer1(inputX,inputY,numKernels,kernelX,kernelY);
    layer1.updateFilters(newfilters);

    std::vector<Matrix> result = layer1.applyConvolution(inputMatrix);
    
    for (size_t k = 0 ; k < result.size(); k++)
    {
        std::vector<intmax_t> resultFlat = result.at(k).flatten();
    
        for (size_t i = 0; i < resultFlat.size(); i++)
        {
            printBits(resultFlat.at(i));
            std::cout << "int value: " << resultFlat.at(i) << std::endl;
        }
        std::cout << std::endl;
    }
    
    //std::cout << (int)layer1.filters[0].numRows();
    
}
