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
#include <chrono>

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

void writeMatrixToCSV(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix.at(i).size(); ++j) {
            outputFile << matrix[i][j];
            if (j != matrix[i].size() - 1) {
                outputFile << ","; // Add comma except for the last element in a row
            }
        }
        outputFile << std::endl; // Move to the next line after each row
    }

    outputFile.close();
    //std::cout << "Matrix written to " << filename << " successfully." << std::endl;
}

int main()
{
    //auto startL2 = std::chrono::steady_clock::now();
    
    std::string layer0Weights = "./weights/layer_0/weights.csv";
    std::string layer0Biases = "./weights/layer_0/biases.csv";
    std::string layer2Weights = "./weights/layer_2/weights.csv";
    std::string layer2Biases = "./weights/layer_2/biases.csv";
    std::string layer4Weights = "./weights/layer_4/weights.csv";
    std::string layer4Biases = "./weights/layer_4/biases.csv";
    std::string layer6Weights = "./weights/layer_6/weights.csv";
    std::string layer6Biases = "./weights/layer_6/biases.csv";
    std::string layer7Weights = "./weights/layer_7/weights.csv";
    std::string layer7Biases = "./weights/layer_7/biases.csv";

    std::string inputBatches = "./weights/batch.csv";
    /*
    
    std::string layer0Weights = "./small_network_weights/layer_0/weights.csv";
    std::string layer0Biases = "./small_network_weights/layer_0/biases.csv";
    std::string layer2Weights = "./small_network_weights/layer_2/weights.csv";
    std::string layer2Biases = "./small_network_weights/layer_2/biases.csv";
    std::string layer4Weights = "./small_network_weights/layer_4/weights.csv";
    std::string layer4Biases = "./small_network_weights/layer_4/biases.csv";
    std::string layer6Weights = "./small_network_weights/layer_6/weights.csv";
    std::string layer6Biases = "./small_network_weights/layer_6/biases.csv";
    std::string layer7Weights = "./small_network_weights/layer_7/weights.csv";
    std::string layer7Biases = "./small_network_weights/layer_7/biases.csv";

    std::string inputBatches = "./small_network_weights/images.csv";*/
   

    ReadParameters layer0(layer0Weights, layer0Biases);
    ReadParameters layer2(layer2Weights, layer2Biases);
    ReadParameters layer4(layer4Weights, layer4Biases);
    ReadParameters layer6(layer6Weights, layer6Biases);
    ReadParameters layer7(layer7Weights, layer7Biases);

    /********************* ARCHITECTURE **************************************/
    ConvolutionalLayer conv2d(16,16,40,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
    PoolingLayer max_pooling2d(2,2);            //(PoolSizeX, PoolSizeY)
    ConvolutionalLayer conv2d_1(7,7,2,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
  //PoolingLayer max_pooling2d(2,2);  *Gentagelse(samme bruges igen)*          //(PoolSizeX, PoolSizeY)
    ConvolutionalLayer conv2d_2(3,3,40,2,2);    //(InputSizeX, InputSizeY, NumberOfKernels, KernelSizeX, KernelSizeY)
    FullyConnectedLayer dense(160,40);          //(InputNodes, OutputNodes)
    FullyConnectedLayer dense1(40,10);         //(InputNodes, OutputNodes)

    /********************* FIXEDPOINT CONVERTER *******************************/
    size_t fracBits = 6;
    FixedPointConverter<intmax_t> converter(6, fracBits);
    FixedPointConverter<double> converter2(6, fracBits); // int type, 4 decimal bits, 4 fractional bits
    FixedPointConverter<intmax_t> converter3(12, 2*fracBits);
    
    /********************* READ INPUT ****************************************/
    std::vector<std::vector<float>> inputBatch = readInput(inputBatches);

    std::vector<std::vector<double>> outputBatch;

    for (int i = 0; i < inputBatch.size(); i++)
    {
        /*  Get the i'th input in fixed point  */
        std::vector<intmax_t> singleInputFixedPoint = converter.convertToFixedPoint(inputBatch.at(i));
        
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
        std::vector<Matrix> FMLayer0beforeRelu = conv2d.applyConvolution(singleInputFixedPointMatrix);
        /************************ APPLY ReLU **********************************************************************************/
        std::vector<Matrix> FMLayer0;
        for ( int j = 0 ; j <  FMLayer0beforeRelu.size(); j++)
        {
            FMLayer0.push_back(FMLayer0beforeRelu.at(j).applyRelu());
        }
        /******************** MAX_POOLING_1*****************************************************/
        std::vector<Matrix> pooledFMLayer0 = max_pooling2d.applyMaxPooling(FMLayer0);
        /************************ TUNCATE FXP **********************************************************************************/
        std::vector<Matrix> FMLayer0out;
        for (int j = 0; j < pooledFMLayer0.size(); j++)
        {
            Matrix truncateMatrixLayer0(pooledFMLayer0.at(j).numRows(),pooledFMLayer0.at(j).numCols());
            truncateMatrixLayer0.unflatten(converter3.truncateLSBs(pooledFMLayer0.at(j).flatten(), fracBits));
            FMLayer0out.push_back(truncateMatrixLayer0);
        }
        /******************* LAYER 2 INSTANTIATE *************************************************************/
        LayerParams ParametersForLayer2 = layer2.getLayer();
        
        /******************** BIAS TO FXP ******************************************************************/
        std::vector<intmax_t> biasesLayer2Fixed = converter.convertToFixedPoint(ParametersForLayer2.biases);
        std::vector<intmax_t> zeroBias(ParametersForLayer2.weights.size(),0);

        std::vector<std::vector<std::vector<float>>> weightsLayer2Inputi(ParametersForLayer2.weights.size());
        std::vector<std::vector<Matrix>> FMLayer2inter;

        for (int j = 0; j < ParametersForLayer2.weights.size(); j++)
        {
            weightsLayer2Inputi.at(j) = PartitionUtility::partitionVector(ParametersForLayer2.weights.at(j), partitionSize);
            std::vector<Matrix> weightsLayer2InputiMatrix;
            for (int p = 0; p < weightsLayer2Inputi.at(j).size(); p++)
            {
                Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer2Inputi.at(j).at(p)));
                weightsLayer2InputiMatrix.push_back(intermediateWeightMatrix);
    
            }
            /******************** RUN LAYER 2 *************************************************/
            conv2d_1.updateFilters(weightsLayer2InputiMatrix,zeroBias);
            FMLayer2inter.push_back(conv2d_1.applyConvolution(FMLayer0out.at(j)));
        }
        /************************* ADD UP ALL FM IN CHANNELS*************************************************************************/
        std::vector<Matrix> FMLayer2beforeRelu;
        for (int p = 0; p < FMLayer2inter.at(0).size(); p++)
        {
            Matrix accumulate(FMLayer2inter.at(0).at(0).numRows(),FMLayer2inter.at(0).at(0).numCols());
            
            for (int j = 0; j < FMLayer2inter.size(); j++)
            {
                accumulate = accumulate + FMLayer2inter.at(j).at(p);
            }
            FMLayer2beforeRelu.push_back(accumulate);
        }
        /************************ APPLY ReLU **********************************************************************************/
        std::vector<Matrix> FMLayer2;
        for ( int j = 0 ; j <  FMLayer2beforeRelu.size(); j++)
        {
            FMLayer2.push_back(FMLayer2beforeRelu.at(j).applyRelu());
        }
        /************************ POOLING NR. 2 ****************************************************************************************/
        std::vector<Matrix> pooledFMLayer2 = max_pooling2d.applyMaxPooling(FMLayer2);
        /************************ TUNCATE FXP **********************************************************************************/
        std::vector<Matrix> FMLayer2out;
        for (int j = 0; j < pooledFMLayer2.size(); j++)
        {
            Matrix truncateMatrixLayer2(pooledFMLayer2.at(j).numRows(),pooledFMLayer2.at(j).numCols());
            truncateMatrixLayer2.unflatten(converter3.truncateLSBs(pooledFMLayer2.at(j).flatten(), fracBits));
            FMLayer2out.push_back(truncateMatrixLayer2);
        }
        /******************* LAYER 4 INSTANTIATE *************************************************************/
        LayerParams ParametersForLayer4 = layer4.getLayer();
        
        /******************** BIAS TO FXP ******************************************************************/
        std::vector<intmax_t> biasesLayer4Fixed = converter.convertToFixedPoint(ParametersForLayer4.biases);

        std::vector<std::vector<std::vector<float>>> weightsLayer4Inputi(ParametersForLayer4.weights.size());
        std::vector<std::vector<Matrix>> FMLayer4inter;
        
        for (int j = 0; j < ParametersForLayer4.weights.size(); j++)
        {
            weightsLayer4Inputi.at(j) = PartitionUtility::partitionVector(ParametersForLayer4.weights.at(j), partitionSize);
            std::vector<Matrix> weightsLayer4InputiMatrix;
            for (int p = 0; p < weightsLayer4Inputi.at(j).size(); p++)
            {
                Matrix intermediateWeightMatrix(2,2,converter.convertToFixedPoint(weightsLayer4Inputi.at(j).at(p)));
                weightsLayer4InputiMatrix.push_back(intermediateWeightMatrix);
            }
            /******************** RUN LAYER 4 *************************************************/          
            conv2d_2.updateFilters(weightsLayer4InputiMatrix,zeroBias);
            FMLayer4inter.push_back(conv2d_2.applyConvolution(FMLayer2out.at(j)));
        }        
        /************************* ADD UP ALL FM IN CHANNELS*************************************************************************/
        std::vector<Matrix> FMLayer4beforeRelu;
        
        for (int p = 0; p < FMLayer4inter.at(0).size(); p++)
            {
            Matrix accumulate(FMLayer4inter.at(0).at(0).numRows(),FMLayer4inter.at(0).at(0).numCols());
            
            for (int j = 0; j < FMLayer4inter.size(); j++)
            {
                accumulate = accumulate + FMLayer4inter.at(j).at(p); 
                accumulate = accumulate + biasesLayer4Fixed.at(j);
            }
            FMLayer4beforeRelu.push_back(accumulate );
        }
        /************************ APPLY ReLU **********************************************************************************/
        std::vector<std::vector<intmax_t>> FMLayer4;
        for ( int j = 0 ; j <  FMLayer4beforeRelu.size(); j++)
        {
            FMLayer4.push_back((FMLayer4beforeRelu.at(j).applyRelu()).flatten());
        }
        /************************* FLATTEN FEATURE MAPS ********************************************************************/
        std::vector<intmax_t> flattenedVector;
        for (int p = 0; p < FMLayer4.size(); p++) 
        {
            flattenedVector.push_back(FMLayer4.at(p).at(0));
        }
        for (int p = 0; p < FMLayer4.size(); p++) 
        {
            flattenedVector.push_back(FMLayer4.at(p).at(2));
        }
        for (int p = 0; p < FMLayer4.size(); p++) 
        {
            flattenedVector.push_back(FMLayer4.at(p).at(1));
        }
        for (int p = 0; p < FMLayer4.size(); p++) 
        {
            flattenedVector.push_back(FMLayer4.at(p).at(3));
        }
        /************************ TUNCATE FXP **********************************************************************************/
        std::vector<intmax_t> flattenedOut = converter3.truncateLSBs(flattenedVector,fracBits);      
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
        std::vector<intmax_t> denseLayer6 = dense.forward(flattenedOut,layer6weightsFixed,layer6biasesFixed);
        /************************ TUNCATE FXP **********************************************************************************/
        std::vector<intmax_t> denseLayer6out = converter3.truncateLSBs(denseLayer6,fracBits);  
        /**************************** DENSE LAYER7 INIT *******************************************************************************/
        LayerParams layer7Params = layer7.getLayer();
        /**************************** FXP CONVERSION *********************************************************************************/
        std::vector<std::vector<intmax_t>> layer7weightsFixed;
        std::vector<intmax_t> layer7biasesFixed = converter.convertToFixedPoint(layer7Params.biases);
        
        for(int j = 0; j < layer7Params.weights.size(); j++)
        {
            std::vector<intmax_t> intermediateDenseWeights = converter.convertToFixedPoint(layer7Params.weights.at(j));
            layer7weightsFixed.push_back(intermediateDenseWeights);
        }
        /*************************** RUN DENSE LAYER7 ********************************************************************************/
        std::vector<intmax_t> denseLayer7 = dense1.forward(denseLayer6out,layer7weightsFixed,layer7biasesFixed);
        /************************ TUNCATE FXP **********************************************************************************/
        std::vector<double> denseLayer7out = converter2.convertToDouble(converter3.truncateLSBs(denseLayer7,fracBits));
        outputBatch.push_back(denseLayer7out);
    }
    writeMatrixToCSV("./weights/output.csv",outputBatch);
    //writeMatrixToCSV("./output2.csv",outputBatch);
}
