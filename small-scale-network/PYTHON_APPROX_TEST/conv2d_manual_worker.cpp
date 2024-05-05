#include <vector>

extern "C" {
    void worker(int i, std::vector<std::vector<std::vector<std::vector<double>>>>& inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& kernel, std::vector<std::vector<std::vector<double>>>& output) {
        int j_dim = inputs[0].size() - kernel[0].size() + 1;
        int k_dim = inputs[0][0].size() - kernel[0][0].size() + 1;
        int l_dim = kernel[0][0][0].size();

        output = std::vector<std::vector<std::vector<double>>>(j_dim, std::vector<std::vector<double>>(k_dim, std::vector<double>(l_dim, 0)));

        for (int j = 0; j < j_dim; ++j) {
            for (int k = 0; k < k_dim; ++k) {
                for (int l = 0; l < l_dim; ++l) {
                    for (int m = 0; m < kernel.size(); ++m) {
                        for (int n = 0; n < kernel[0].size(); ++n) {
                            for (int o = 0; o < inputs[0][0][0].size(); ++o) {
                                output[j][k][l] += inputs[i][j+m][k+n][o] * kernel[m][n][o][l];
                            }
                        }
                    }
                }
            }
        }
    }
}