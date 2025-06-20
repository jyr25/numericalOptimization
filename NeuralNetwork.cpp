#include "NeuralNetwork.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// ------------------------
// 생성자
// ------------------------
NeuralNetwork::NeuralNetwork(int input, int output, int hiddenLayers, int units, double lr)
    : inputSize(input), outputSize(output), hiddenLayers(hiddenLayers), unitsPerLayer(units), learningRate(lr) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int totalLayers = hiddenLayers + 1;

    for (int l = 0; l < totalLayers; l++) {
        int from = (l == 0) ? inputSize : unitsPerLayer;
        int to = (l == hiddenLayers) ? outputSize : unitsPerLayer;

        std::vector<std::vector<double>> layerWeights(from, std::vector<double>(to));
        for (int i = 0; i < from; i++) {
            for (int j = 0; j < to; j++) {
                layerWeights[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5;
            }
        }
        weights.push_back(layerWeights);
    }

    for (int l = 0; l < totalLayers; l++) {
        int size = (l == hiddenLayers) ? outputSize : unitsPerLayer;
        std::vector<double> layerBiases(size);
        for (int i = 0; i < size; i++) {
            layerBiases[i] = static_cast<double>(rand()) / RAND_MAX - 0.5;
        }
        biases.push_back(layerBiases);
    }
}

// ------------------------
// 활성화 함수 및 미분
// ------------------------
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// ------------------------
// 순전파 (Forward)
// ------------------------
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> activation = input;

    for (int l = 0; l < hiddenLayers + 1; l++) {
        int nextSize = (l == hiddenLayers) ? outputSize : unitsPerLayer;
        std::vector<double> nextActivation(nextSize, 0.0);

        for (int j = 0; j < nextSize; j++) {
            for (int i = 0; i < activation.size(); i++) {
                nextActivation[j] += activation[i] * weights[l][i][j];
            }
            nextActivation[j] += biases[l][j];
            nextActivation[j] = sigmoid(nextActivation[j]);
        }

        activation = nextActivation;
    }

    return activation;
}

// ------------------------
// 손실 함수 (MSE)
// ------------------------
double NeuralNetwork::loss(const std::vector<double>& y, const std::vector<double>& y_hat) {
    double sum = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        sum += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }
    return sum / y.size();
}

// ------------------------
// 학습 함수 (Backpropagation)
// ------------------------
void NeuralNetwork::train(const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (size_t sample = 0; sample < X.size(); sample++) {
            const std::vector<double>& input = X[sample];
            const std::vector<double>& target = y[sample];

            // [1] 순전파
            std::vector<std::vector<double>> activations;
            std::vector<std::vector<double>> zs;

            std::vector<double> a = input;
            activations.push_back(a);

            for (int l = 0; l < hiddenLayers + 1; l++) {
                int toSize = (l == hiddenLayers) ? outputSize : unitsPerLayer;
                std::vector<double> z(toSize, 0.0);
                std::vector<double> next_a(toSize, 0.0);

                for (int j = 0; j < toSize; j++) {
                    for (int i = 0; i < a.size(); i++) {
                        z[j] += a[i] * weights[l][i][j];
                    }
                    z[j] += biases[l][j];
                    next_a[j] = sigmoid(z[j]);
                }

                zs.push_back(z);
                activations.push_back(next_a);
                a = next_a;
            }

            // [2] 출력층 델타
            std::vector<std::vector<double>> deltas(hiddenLayers + 1);
            int L = hiddenLayers;

            std::vector<double> delta_output(outputSize, 0.0);
            for (int k = 0; k < outputSize; k++) {
                double z = zs[L][k];
                delta_output[k] = (activations[L + 1][k] - target[k]) * sigmoid_derivative(z);
            }
            deltas[L] = delta_output;

            // [3] 은닉층 델타
            for (int l = L - 1; l >= 0; l--) {
                int size = unitsPerLayer;
                std::vector<double> delta(size, 0.0);

                for (int i = 0; i < size; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < deltas[l + 1].size(); j++) {
                        sum += weights[l + 1][i][j] * deltas[l + 1][j];
                    }
                    delta[i] = sum * sigmoid_derivative(zs[l][i]);
                }

                deltas[l] = delta;
            }

            // [4] 가중치 & 편향 업데이트
            for (int l = 0; l < hiddenLayers + 1; l++) {
                for (int i = 0; i < weights[l].size(); i++) {
                    for (int j = 0; j < weights[l][i].size(); j++) {
                        weights[l][i][j] -= learningRate * activations[l][i] * deltas[l][j];
                    }
                }
                for (int j = 0; j < biases[l].size(); j++) {
                    biases[l][j] -= learningRate * deltas[l][j];
                }
            }

            totalLoss += loss(target, activations.back());
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / X.size() << std::endl;
    }

}
