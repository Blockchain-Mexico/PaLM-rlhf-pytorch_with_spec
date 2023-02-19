#ifndef AUTOREGRESSIVE_TRANSFORMER_H
#define AUTOREGRESSIVE_TRANSFORMER_H

#include <vector>

class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size);

    std::vector<double> forward(std::vector<double> x);

    void backward(std::vector<double> x, std::vector<double> y_true, std::vector<double> y_pred);

private:
    std::vector<std::vector<double>> W1, W2, dW1, dW2;
    std::vector<double> b1, b2, db1, db2, h1;

    static double relu(double x);
    static std::vector<double> softmax(const std::vector<double>& x);

    static const double learning_rate;
};

#endif
