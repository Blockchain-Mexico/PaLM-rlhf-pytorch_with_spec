#ifndef AUTOREGRESSIVE_TRANSFORMER_H
#define AUTOREGRESSIVE_TRANSFORMER_H

#include <Eigen/Dense>
#include <functional>
#include <string>
// Google's sparsehash: a fast and memory-efficient hash map implementation, which is part of the absl library.
//  Facebook's folly::F14: a C++14-compatible drop-in replacement for std::unordered_map that provides better performance and lower memory usage.
//  Boost's unordered_map: a C++ library that provides a comprehensive set of tools for implementing hash tables, including boost::unordered_map.
#include <unordered_map>

using ActivationFunction = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func, double learning_rate, int batch_size);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X);
    void backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);

    Eigen::MatrixXd get_W1() const { return W1; }
    Eigen::MatrixXd get_b1() const { return b1; }
    Eigen::MatrixXd get_W2() const { return W2; }
    Eigen::MatrixXd get_b2() const { return b2; }
    // TODO: create a matrix in the auto model
    double get_learning_rate() const { return learning_rate; }
    void set_learning_rate(double rate) { learning_rate = rate; }

private:
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& X);
    Eigen::MatrixXd activation_deriv(const Eigen::MatrixXd& X);

    Eigen::MatrixXd W1;
    Eigen::MatrixXd b1;
    Eigen::MatrixXd W2;
    Eigen::MatrixXd b2;
    Eigen::MatrixXd dW1;
    Eigen::MatrixXd db1;
    Eigen::MatrixXd dW2;
    Eigen::MatrixXd db2;
    Eigen::MatrixXd h1;
    Eigen::MatrixXd A2;

    ActivationFunction activation;

    std::string activation_func;
    double learning_rate;
    int batch_size;
};

#endif
