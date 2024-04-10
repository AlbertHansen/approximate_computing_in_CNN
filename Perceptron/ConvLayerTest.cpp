#include <iostream>
#include "Perceptron.h"
#include "ConvolutionalLayer.h" 
#include "Matrix.h"
#include "Adder.h"
#include <vector>

int main()
{
    
    std::vector<intmax_t> flattenedInput =   {1,3,2,5,2,
                                            1,2,5,4,1,
                                            1,2,3,4,5,
                                            5,2,1,4,4,
                                            1,1,2,2,3};
    
    std::vector<intmax_t> flattenedFilter1 =   {1,-2,-3,
                                              -3,1,2,
                                              -3,1,1};
    std::vector<intmax_t> flattenedFilter2 =   {1,1,1,
                                              1,1,1,
                                              1,1,1};
    
    Matrix inputMatrix(5,5,flattenedInput);
    Matrix filterMatrix1(3,3,flattenedFilter1);
    Matrix filterMatrix2(3,3,flattenedFilter2);

    size_t inputX = inputMatrix.numCols();
    size_t inputY = inputMatrix.numRows();
    size_t numKernels = 2;
    size_t kernelX = filterMatrix1.numCols();
    size_t kernelY = filterMatrix1.numRows();

    ConvolutionalLayer layer1(inputX,inputY,numKernels,kernelX,kernelY);

    std::vector<Matrix> newfilters = {filterMatrix1,filterMatrix2};

    layer1.updateFilters(newfilters);

    std::vector<intmax_t> w = {-2,-2};
    std::vector<intmax_t> I = {2,2};

    intmax_t a = -2;
    intmax_t b = -2;

    Perceptron percept(w,I);
    Adder adder;
    std::cout << "ADD: " << adder.add(a,b) << std::endl;
    std::cout << "PERCEPT: " << percept.compute(0) << std::endl;
    /*
    std::vector<Matrix> result = layer1.applyConvolution(inputMatrix);
    for (size_t k = 0 ; k < result.size(); k++)
    {
        std::vector<intmax_t> resultFlat = result.at(k).flatten();
    
        for (size_t i = 0; i < resultFlat.size(); i++)
        {
            std::cout << resultFlat.at(i) << " Size: " << sizeof(resultFlat.at(i)) << " ";
        }
        std::cout << std::endl;
    }*/
    //std::cout << (int)layer1.filters[0].numRows();
    
}
