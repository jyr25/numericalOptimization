#include <vector>

class NeuralNetwork {
private:
    int inputSize;
    int outputSize;
    int hiddenLayers;
    int unitsPerLayer;
    double learningRate;

    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    // 내부에서만 사용하는 함수들
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double loss(const std::vector<double>& y, const std::vector<double>& y_hat);

public:
    NeuralNetwork(int input, int output, int hiddenLayers, int units, double lr);

    std::vector<double> forward(const std::vector<double>& input);

    void train(const std::vector<std::vector<double>>& X,
        const std::vector<std::vector<double>>& y,
        int epochs);
};

#endif // NEURAL_NETWORK_H
