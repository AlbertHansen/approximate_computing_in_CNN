#include <vector>

std::vector<std::vector<std::vector<std::vector<float>>>> conv2d_manual(const std::vector<std::vector<std::vector<std::vector<float>>>>& inputs, const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel) {
    int batchSize = inputs.size();
    int inputHeight = inputs.at(0).size();
    int inputWidth = inputs.at(0).at(0).size();
    int inputChannels = inputs.at(0).at(0).at(0).size();

    int kernelHeight = kernel.size();
    int kernelWidth = kernel.at(0).size();
    int kernelInChannels = kernel.at(0).at(0).size();
    int kernelOutChannels = kernel.at(0).at(0).at(0).size();

    std::vector<std::vector<std::vector<std::vector<float>>>> output(batchSize, std::vector<std::vector<std::vector<float>>>(inputHeight - kernelHeight + 1, std::vector<std::vector<float>>(inputWidth - kernelWidth + 1, std::vector<float>(kernelOutChannels, 0))));

    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < inputHeight - kernelHeight + 1; ++j) {
            for (int k = 0; k < inputWidth - kernelWidth + 1; ++k) {
                for (int l = 0; l < kernelOutChannels; ++l) {
                    for (int m = 0; m < kernelHeight; ++m) {
                        for (int n = 0; n < kernelWidth; ++n) {
                            for (int o = 0; o < inputChannels; ++o) {
                                output.at(i).at(j).at(k).at(l) += inputs.at(i).at(j + m).at(k + n).at(o) * kernel.at(m).at(n).at(o).at(l);
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}
