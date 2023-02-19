#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return std::max(0.0, x);
}

double elu(double x) {
    const double alpha = 1.0;
    return (x >= 0) ? x : alpha * (exp(x) - 1);
}

double leaky_relu(double x) {
    const double alpha = 0.01;
    return (x >= 0) ? x : alpha * x;
}

// Softmax function
std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> exps(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum_exps = 0.0;
    for (int i = 0; i < x.size(); i++) {
        exps[i] = exp(x[i] - max_val);
        sum_exps += exps[i];
    }
    for (int i = 0; i < x.size(); i++) {
        exps[i] /= sum_exps;
    }
    return exps;
}

// AutoregressiveTransformer class
class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func)
        : W1(hidden_size, std::vector<double>(input_size)),
          b1(hidden_size),
          W2(output_size, std::vector<double>(hidden_size)),
          b2(output_size),
          dW1(hidden_size, std::vector<double>(input_size)),
          db1(hidden_size),
          dW2(output_size, std::vector<double>(hidden_size)),
          db2(output_size),
          h1(hidden_size),
          activation_func(activation_func) {
        // Initialize the weights and biases using Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev1 = sqrt(2.0 / (input_size + hidden_size));
        double std_dev2 = sqrt(2.0 / (hidden_size + output_size));
        std::normal_distribution<double> dist1(0, std_dev1);
        std::normal_distribution<double> dist2(0, std_dev2);
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W1[i][j] = dist1(gen);
            }
            b1[i] = 0.0;
        }
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                W2[i][j] = dist2(gen);
            }
            b2[i] = 0.0;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) {
        // implement the parralellism
        // Calculate the first layer activations
        for (int i = 0; i < W1.size(); i++) {
            h1[i] = b1[i];
            for (int j = 0; j < x.size(); j++) {
                h1[i] += W1[i][j] * x[j];
            }
            h1[i] = activation(h1[i]);
        }

        // Calculate the second layer activations
        std::vector<double> h2(b2.size());

