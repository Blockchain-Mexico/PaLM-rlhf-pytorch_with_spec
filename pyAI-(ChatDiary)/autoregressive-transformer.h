#ifndef AUTOREGRESSIVE_TRANSFORMER_H
#define AUTOREGRESSIVE_TRANSFORMER_H

#include <vector>
#include <string>

class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size, std::string activation_func);
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs);
    std::vector<double> forward(const std::vector<double>& x);

private:
    double batch_size = 32;
    double learning_rate = 0.001;

    Eigen::MatrixXd W1;
    Eigen::VectorXd b1;
    Eigen::MatrixXd W2;
    Eigen::VectorXd b2;

    Eigen::MatrixXd dW1;
    Eigen::VectorXd db1;
    Eigen::MatrixXd dW2;
    Eigen::VectorXd db2;

    Eigen::VectorXd h1;
    Eigen::VectorXd h2;

    std::function<double(double)> activation;
    std::string activation_func;

    void backward(const std::vector<double>& x, const std::vector<double>& grad);
    void update_parameters();
};

#endif // AUTOREGRESSIVE_TRANSFORMER_H
