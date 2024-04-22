#include <iostream>
#include "Perceptron.h"
#include "ConvolutionalLayer.h" 
#include "Matrix.h"
#include "Adder.h"
#include <vector>
#include "FixedPointConverter.h"
#include "PoolingLayer.h"

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
    

    FixedPointConverter<int> converter(4, 4); // int type, 4 decimal bits, 4 fractional bits

    
    std::vector<int> fixedInput = converter.convertToFixedPoint(flattenedInput);
    std::vector<int> fixedFilt1 = converter.convertToFixedPoint(flattenedFilter1);
    std::vector<int> fixedFilt2 = converter.convertToFixedPoint(flattenedFilter2);    
    
    
    Matrix inputMatrix(5,5,fixedInput);
    Matrix filterMatrix1(3,3,fixedFilt1);
    Matrix filterMatrix2(3,3,fixedFilt2);
    std::vector<Matrix> newfilters = {filterMatrix1,filterMatrix2};

    size_t inputX = inputMatrix.numCols();
    size_t inputY = inputMatrix.numRows();
    size_t numKernels = 40;
    size_t kernelX = filterMatrix1.numCols();
    size_t kernelY = filterMatrix1.numRows();

    ConvolutionalLayer conv2d(inputX,inputY,numKernels,kernelX,kernelY);
    PoolingLayer max_pooling2d(2,2);
    ConvolutionalLayer conv2d_1(7,7,numKernels,kernelX,kernelY); 
    ConvolutionalLayer conv2d_1(7,7,numKernels,kernelX,kernelY); 
    conv2d.updateFilters(newfilters);

    std::vector<Matrix> result = conv2d.applyConvolution(inputMatrix);

    std::vector<Matrix> poolOut = poolLayer1.applyMaxPooling(result);

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
    
    for (size_t k = 0 ; k < result.size(); k++)
    {
        std::vector<intmax_t> resultFlat = poolOut.at(k).flatten();
    
        for (size_t i = 0; i < resultFlat.size(); i++)
        {
            printBits(resultFlat.at(i));
            std::cout << "int value: " << resultFlat.at(i) << std::endl;
        }
        std::cout << std::endl;
    }
    
    //std::cout << (int)layer1.filters[0].numRows();
    
}
