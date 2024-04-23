#include <iostream>
#include "Perceptron.h"
#include "ConvolutionalLayer.h" 
#include "Matrix.h"
#include "Adder.h"
#include <vector>
#include "FixedPointConverter.h"
#include "FullyConnectedLayer.h"
#include "PoolingLayer.h"
#include "ReadParameters.h"
#include <fstream>
#include <sstream>
#include <string>

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

std::vector<std::vector<float>> readInput(const std::string& filename) {
    std::vector<std::vector<float>> data;

    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;  // Return empty data
    }

    // Read each line from the file
    std::string line;
    while (getline(file, line)) {
        // Create a stringstream from the line
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        // Read each cell from the stringstream
        while (getline(ss, cell, ',')) {
            // Convert cell to float
            row.push_back(stof(cell));
        }

        // Add the row to the data vector
        data.push_back(row);
    }

    // Close the file
    file.close();

    return data;
}

int main()
{
    std::string layer0Weights = "WeightsNBiases/before/layer_0/weights.csv";
    std::string layer0Biases = "WeightsNBiases/before/layer_0/biases.csv";
    std::string layer2Weights = "WeightsNBiases/before/layer_2/weights.csv";
    std::string layer2Biases = "WeightsNBiases/before/layer_2/biases.csv";
    std::string layer4Weights = "WeightsNBiases/before/layer_4/weights.csv";
    std::string layer4Biases = "WeightsNBiases/before/layer_4/biases.csv";
    std::string layer6Weights = "WeightsNBiases/before/layer_6/weights.csv";
    std::string layer6Biases = "WeightsNBiases/before/layer_6/biases.csv";
    std::string layer7Weights = "WeightsNBiases/before/layer_7/weights.csv";
    std::string layer7Biases = "WeightsNBiases/before/layer_7/biases.csv";

    std::string inputBatches = "WeightsNBiases/batch_test.csv";

    ReadParameters layer0(layer0Weights, layer0Biases);
    ReadParameters layer2(layer2Weights, layer2Biases);
    ReadParameters layer4(layer4Weights, layer4Biases);
    ReadParameters layer6(layer6Weights, layer6Biases);
    ReadParameters layer7(layer7Weights, layer7Biases);

    /********************* ARCHITECTURE **************************************/
    ConvolutionalLayer conv2d(16,16,40,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
    PoolingLayer max_pooling2d(2,2);            //(PoolSizeX, PoolSizeY)
    ConvolutionalLayer conv2d_1(7,7,40,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
  //PoolingLayer max_pooling2d(2,2);  *Gentagelse(samme bruges igen)*          //(PoolSizeX, PoolSizeY)
    ConvolutionalLayer conv2d_2(3,3,40,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
    FullyConnectedLayer dense(160,40);          //(InputNodes, OutputNodes)
    FullyConnectedLayer dense1(40,100);         //(InputNodes, OutputNodes)

    /********************* FIXEDPOINT CONVERTER *******************************/
    FixedPointConverter<intmax_t> converter(4, 4); // int type, 4 decimal bits, 4 fractional bits
    /********************* READ INPUT ****************************************/
    std::vector<std::vector<float>> inputBatch = readInput(inputBatches);

    for (int i = 0; i < 1/*inputBatch.size()*/; i++)
    {
        /*  Get the i'th input in fixed point  */
        std::vector<intmax_t> singleInputFixedPoint =  converter.convertToFixedPoint(inputBatch.at(i));
        Matrix singleInputFixedPointMatrix(16,16,singleInputFixedPoint);
        /********************* LAYER 0 INSTANTIATE *******************************************/
        size_t partitionSize = 4;
        LayerParams ParametersForLayer0 = layer0.getLayer();
        std::vector<std::vector<float>> weightsLayer0Inputi = PartitionUtility::partitionVector(ParametersForLayer0.weights.at(i), partitionSize);
        
        std::vector<Matrix> weightsLayer0InputiMatrix;
        for (int j = 0; j < weightsLayer0Inputi.size(); j++)
        {
            //Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer0Inputi.at(j)));
            //weightsLayer0InputiMatrix.push_back(intermediateWeightMatrix);
        }
        /******************** RUN LAYER 0 *************************************************/
        conv2d.updateFilters(weightsLayer0InputiMatrix);
        //std::vector<Matrix> FMLayer0 = conv2d.applyConvolution(singleInputFixedPointMatrix);
        //for ( int p = 0; p < FMLayer0.size(); p++)
        {
            //std::cout << FMLayer0.at(0)(1,1) << FMLayer0.at(0)(1,2) << std::endl << FMLayer0.at(0)(2,1) << FMLayer0.at(0)(2,2);
        }
        /******************** MAX_POOLING_1*****************************************************/
        //std::vector<Matrix> pooledFMLayer0 = max_pooling2d.applyMaxPooling(FMLayer0);
    }
    


    LayerParams testLayer = layer0.getLayer();

    //std::cout << testLayer.weights.at(39).at(3) << std::endl;

    
    
    /*
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
    std::vector<float> flatBias         =   {0.0,0.0};
    

    FixedPointConverter<intmax_t> converter(4, 4); // int type, 4 decimal bits, 4 fractional bits

    
    std::vector<intmax_t> fixedInput = converter.convertToFixedPoint(flattenedInput);
    std::vector<intmax_t> fixedFilt1 = converter.convertToFixedPoint(flattenedFilter1);
    std::vector<intmax_t> fixedFilt2 = converter.convertToFixedPoint(flattenedFilter2);   
    std::vector<intmax_t> fixedBias = converter.convertToFixedPoint(flatBias);  
        /*intmax_t inter = 0;
        for (int i = 0; i < fixedInput.size(); i++)
        {
            std::cout << "Input: " << fixedInput.at(i) << std::endl;
            printBits(fixedInput.at(i));
            inter += fixedInput.at(i);
            std::cout << "inter: " << inter << std::endl;
        }
    
        std::vector<std::vector<intmax_t>> weights = {fixedFilt1,fixedFilt2};

        FullyConnectedLayer dense1(fixedInput,weights); 

        std::vector<intmax_t> result = dense1.forward(fixedBias);

        for (int i = 0; i < result.size(); i++)
        {
            std::cout << result.at(i) << std::endl;
            printBits(result.at(i));
        }

    
    Matrix inputMatrix(5,5,fixedInput);
    Matrix filterMatrix1(3,3,fixedFilt1);
    Matrix filterMatrix2(3,3,fixedFilt2);
    std::vector<Matrix> newfilters = {filterMatrix1,filterMatrix2};

    size_t inputX = inputMatrix.numCols();
    size_t inputY = inputMatrix.numRows();
    size_t numKernels = 2;
    size_t kernelX = filterMatrix1.numCols();
    size_t kernelY = filterMatrix1.numRows();

    ConvolutionalLayer conv2d(inputX,inputY,numKernels,kernelX,kernelY);
    PoolingLayer max_pooling2d(2,2);
    ConvolutionalLayer conv2d_1(7,7,numKernels,kernelX,kernelY); 
    ConvolutionalLayer conv2d_2(7,7,numKernels,kernelX,kernelY); 
    conv2d.updateFilters(newfilters);

    std::vector<Matrix> result = conv2d.applyConvolution(inputMatrix);

    std::vector<Matrix> poolOut = max_pooling2d.applyMaxPooling(result);

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
    }*/
    
    //std::cout << (int)layer1.filters[0].numRows();
    
}
