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
            /******************** BIAS TO FXP ******************************************************************/
        std::vector<intmax_t> biasesLayer0Fixed = converter.convertToFixedPoint(ParametersForLayer0.biases);
        std::vector<std::vector<float>> weightsLayer0Inputi;
        for (int j = 0; j < ParametersForLayer0.weights.size(); j++)
        {
            weightsLayer0Inputi = PartitionUtility::partitionVector(ParametersForLayer0.weights.at(j), partitionSize);
        }
        

        std::vector<Matrix> weightsLayer0InputiMatrix;
        for (int j = 0; j < weightsLayer0Inputi.size(); j++)
        {
            //std::cout << "W: " << weightsLayer0Inputi.at(j).at(0) << std::endl;
            Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer0Inputi.at(j)));
            weightsLayer0InputiMatrix.push_back(intermediateWeightMatrix);
        }
        /******************** RUN LAYER 0 *************************************************/
        conv2d.updateFilters(weightsLayer0InputiMatrix, biasesLayer0Fixed);
        std::vector<Matrix> FMLayer0 = conv2d.applyConvolution(singleInputFixedPointMatrix);
        
        /******************** MAX_POOLING_1*****************************************************/
        std::vector<Matrix> pooledFMLayer0 = max_pooling2d.applyMaxPooling(FMLayer0);

        std::cout << pooledFMLayer0.at(0).numRows() << std::endl;
        /******************* LAYER 2 INSTANTIATE *************************************************************/
        LayerParams ParametersForLayer2 = layer2.getLayer();
        
        /******************** BIAS TO FXP ******************************************************************/
        std::vector<intmax_t> biasesLayer2Fixed = converter.convertToFixedPoint(ParametersForLayer2.biases);
        std::vector<intmax_t> zeroBias(ParametersForLayer2.weights.size(),0);

        std::vector<std::vector<std::vector<float>>> weightsLayer2Inputi(ParametersForLayer2.weights.size());
        std::vector<std::vector<Matrix>> FMLayer2inter;
        for (int j = 0; j < ParametersForLayer2.weights.size(); j++)
        {
            //std::cout << j << std::endl;
            weightsLayer2Inputi.at(j) = PartitionUtility::partitionVector(ParametersForLayer2.weights.at(j), partitionSize);
        
            //std::cout << weightsLayer2Inputi.at(j).at(j).size() << std::endl;

            std::vector<Matrix> weightsLayer2InputiMatrix;
            for (int p = 0; p < weightsLayer2Inputi.at(j).size(); p++)
            {
                //std::cout << "W: " << weightsLayer2Inputi.at(j).at(1).at(0) << std::endl;
                Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer2Inputi.at(j).at(p)));
                weightsLayer2InputiMatrix.push_back(intermediateWeightMatrix);
            }
            /******************** RUN LAYER 2 *************************************************/
            if (j== 0)
            {
                conv2d_1.updateFilters(weightsLayer2InputiMatrix, biasesLayer2Fixed);
            } 
            else
            {
                conv2d_1.updateFilters(weightsLayer2InputiMatrix,zeroBias);
            }
            FMLayer2inter.push_back(conv2d_1.applyConvolution(pooledFMLayer0.at(j)));
        }
        /************************* ADD UP ALL FM IN CHANNELS*************************************************************************/
        std::vector<Matrix> FMLayer2;
        for (int j = 0; j < FMLayer2inter.size(); j++)
        {
            Matrix accumulate(FMLayer2inter.at(j).at(0).numRows(),FMLayer2inter.at(j).at(0).numCols());
            
            for (int p = 0; p < FMLayer2inter.at(j).size(); p++)
            {
                accumulate = accumulate + FMLayer2inter.at(j).at(p);
            }
            FMLayer2.push_back(accumulate);
        }
        /************************ POOLING NR. 2 ****************************************************************************************/
        std::vector<Matrix> pooledFMLayer2 = max_pooling2d.applyMaxPooling(FMLayer2);

        std::cout << pooledFMLayer2.at(0).numRows() << std::endl;
        //std::cout << FMLayer2inter.size() << std::endl;
        /******************* LAYER 4 INSTANTIATE *************************************************************/
        LayerParams ParametersForLayer4 = layer4.getLayer();
        
        /******************** BIAS TO FXP ******************************************************************/
        std::vector<intmax_t> biasesLayer4Fixed = converter.convertToFixedPoint(ParametersForLayer4.biases);

        std::vector<std::vector<std::vector<float>>> weightsLayer4Inputi(ParametersForLayer4.weights.size());
        std::vector<std::vector<Matrix>> FMLayer4inter;
        for (int j = 0; j < ParametersForLayer4.weights.size(); j++)
        {
            //std::cout << j << std::endl;
            weightsLayer4Inputi.at(j) = PartitionUtility::partitionVector(ParametersForLayer4.weights.at(j), partitionSize);
        
            //std::cout << weightsLayer2Inputi.at(j).at(j).size() << std::endl;

            std::vector<Matrix> weightsLayer4InputiMatrix;
            for (int p = 0; p < weightsLayer4Inputi.at(j).size(); p++)
            {
                //std::cout << "W: " << weightsLayer2Inputi.at(j).at(1).at(0) << std::endl;
                Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer4Inputi.at(j).at(p)));
                weightsLayer4InputiMatrix.push_back(intermediateWeightMatrix);
            }
            /******************** RUN LAYER 2 *************************************************/
            if (j== 0)
            {
                conv2d_2.updateFilters(weightsLayer4InputiMatrix, biasesLayer4Fixed);
            } 
            else
            {
                conv2d_2.updateFilters(weightsLayer4InputiMatrix,zeroBias);
            }
            FMLayer4inter.push_back(conv2d_2.applyConvolution(pooledFMLayer2.at(j)));
        }
        /************************* ADD UP ALL FM IN CHANNELS*************************************************************************/
        std::vector<std::vector<intmax_t>> FMLayer4;
        for (int j = 0; j < FMLayer4inter.size(); j++)
        {
            Matrix accumulate(FMLayer4inter.at(j).at(0).numRows(),FMLayer4inter.at(j).at(0).numCols());
            
            for (int p = 0; p < FMLayer4inter.at(j).size(); p++)
            {
                accumulate = accumulate + FMLayer4inter.at(j).at(p);
            }
            FMLayer4.push_back(accumulate.flatten());
        }
        /************************* FLATTEN FEATURE MAPS ********************************************************************/
        size_t totalSize = 0;
        for (const auto& innerVector : FMLayer4) {
            totalSize += innerVector.size();
        }
        
        // Flatten the nested vector into a single vector using for loops
        std::vector<intmax_t> flattenedVector;
        flattenedVector.reserve(totalSize);  // Reserve space for efficiency

        for (const auto& innerVector : FMLayer4) 
        {
            for (const auto& element : innerVector) 
            {
                flattenedVector.push_back(element);
            }
        }
        /**************************** DENSE LAYER6 INIT *******************************************************************************/
        LayerParams layer6Params = layer6.getLayer();
        /**************************** FXP CONVERSION *********************************************************************************/
        std::vector<std::vector<intmax_t>> layer6weightsFixed;
        std::vector<intmax_t> layer6biasesFixed = converter.convertToFixedPoint(layer6Params.biases);
        for(int j = 0; j < layer6Params.weights.size(); j++)
        {
            std::vector<intmax_t> intermediateDenseWeights = converter.convertToFixedPoint(layer6Params.weights.at(j));
            layer6weightsFixed.push_back(intermediateDenseWeights);
        }
        /*************************** RUN DENSE LAYER6 ********************************************************************************/
        std::vector<intmax_t> denseLayer6 = dense.forward(flattenedVector,layer6weightsFixed,layer6biasesFixed);
        /**************************** DENSE LAYER7 INIT *******************************************************************************/
        LayerParams layer7Params = layer7.getLayer();
        /**************************** FXP CONVERSION *********************************************************************************/
        std::vector<std::vector<intmax_t>> layer7weightsFixed;
        std::vector<intmax_t> layer7biasesFixed = converter.convertToFixedPoint(layer7Params.biases);
        std::cout << layer7biasesFixed.size();
        for(int j = 0; j < layer7Params.weights.size(); j++)
        {
            std::vector<intmax_t> intermediateDenseWeights = converter.convertToFixedPoint(layer7Params.weights.at(j));
            layer7weightsFixed.push_back(intermediateDenseWeights);
        }
        /*************************** RUN DENSE LAYER7 ********************************************************************************/
        std::vector<intmax_t> denseLayer7 = dense1.forward(denseLayer6,layer7weightsFixed,layer7biasesFixed);

        for (const auto& element : denseLayer7)
        {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
    

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
