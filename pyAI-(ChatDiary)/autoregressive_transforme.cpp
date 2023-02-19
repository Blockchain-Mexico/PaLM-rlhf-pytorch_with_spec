#include <cmath>
#include <vector>

using std::vector;

class AutoregressiveTransformer {
public:
    AutoregressiveTransformer(int input_size, int hidden_size, int output_size)
        : W1(hidden_size, vector<double>(input_size)),
          b1(hidden_size),
          W2(output_size, vector<double>(hidden_size)),
          b2(output_size),
          dW1(hidden_size, vector<double>(input_size)),
          db1(hidden_size),
          dW2(output_size, vector<double>(hidden_size)),
          db2(output_size) {}

    vector<double> forward(vector<double> x) {
        // Calculate the first layer activations
        vector<double> h1(b1.size());
        for (int i = 0; i < W1.size(); i++) {
            for (int j = 0; j < x.size(); j++) {
                h1[i] += W1[i][j] * x[j];
            }
            h1[i] += b1[i];
            h1[i] = relu(h1[i]);
        }

        // Calculate the second layer activations
        vector<double> h2(b2.size());
        for (int i = 0; i < W2.size(); i++) {
            for (int j = 0; j < h1.size(); j++) {
                h2[i] += W2[i][j] * h1[j];
            }
            h2[i] += b2[i];
        }

        // Apply the softmax function to the outputs
        vector<double> y = softmax(h2);

        return y;
    }

    void backward(vector<double> x, vector<double> y_true, vector<double> y_pred) {
        // Calculate the second layer gradients
        for (int i = 0; i < W2.size(); i++) {
            for (int j = 0; j < W2[i].size(); j++) {
                dW2[i][j] = (y_pred[i] - y_true[i]) * h1[j];
            }
        }

        // Calculate the first layer gradients
        for (int i = 0; i < W1.size(); i++) {
            for (int j = 0; j < W1[i].size(); j++) {
                double sum = 0;
                for (int k = 0; k < W2.size(); k++) {
                    sum += (y_pred[k] - y_true[k]) * W2[k][i];
                }
                dW1[i][j] = x[j] * sum * (1 - h1[i]) * h1[i];
            }
        }

        // Update the weights and biases
        for (int i = 0; i < W1.size(); i++) {
            for (int j = 0; j < W1[i].size(); j++) {
                W1[i][j] -= dW1[i][j] * learning_rate;
            }
            b1[i] -= db1[i] * learning_rate;
        }

        for (int i = 0; i < W2.size(); i++) {
            for (int j = 0; j < W2[i].size(); j++) {
                W2[i][j] -= dW2[i][j] * learning_rate;
            }
            b2[i] -= db2[i] * learning_rate;
        }
    }

private:
    vector<vector<double>> W1, W2, dW1, dW2;
    vector<double> b1, b2, db1, db2;
    vector<double> h1;


