#include "autoregressive_transformer.h"
#include <Eigen/Dense>
#include <unordered_map>

AutoregressiveTransformer::AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func, double learning_rate, int batch_size)
    : W1(hidden_size, input_size),
      b1(hidden_size),
      W2(output_size, hidden_size),
      b2(output_size),
      dW1(hidden_size, input_size),
      db1(hidden_size),
      dW2(output_size, hidden_size),
      db2(output_size),
      h1(hidden_size),
      activation_func(activation_func),
      learning_rate(learning_rate),
      batch_size(batch_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev1 = sqrt(2.0 / (input_size + hidden_size));
    double std_dev2 = sqrt(2.0 / (hidden_size + output_size));
    std::normal_distribution<double> dist1(0, std_dev1);
    std::normal_distribution<double> dist2(0, std_dev2);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            W1(i, j) = dist1(gen);
        }
        b1(i) = 0.0;
    }
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            W2(i, j) = dist2(gen);
        }
        b2(i) = 0.0;
    }

    std::unordered_map<std::string, ActivationFunction> activation_map = {
        {"relu", relu},
        {"sigmoid", sigmoid},
        {"elu", elu},
        {"leaky_relu", leaky_relu}
    };
    if (activation_map.find(activation_func) == activation_map.end()) {
        throw std::invalid_argument("Invalid activation function");
    }
    activation = activation
