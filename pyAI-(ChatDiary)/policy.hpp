#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

int policy(const std::vector<double>& q_values, double temperature) {
    // Apply temperature scaling to the Q-values
    std::vector<double> scaled_q_values(q_values.size());
    std::transform(q_values.begin(), q_values.end(), scaled_q_values.begin(),
                   [=](double x) { return std::exp(x / temperature); });

    // Normalize the scaled Q-values
    double total = std::accumulate(scaled_q_values.begin(), scaled_q_values.end(), 0.0);
    std::vector<double> probabilities(q_values.size());
    std::transform(scaled_q_values.begin(), scaled_q_values.end(), probabilities.begin(),
                   [=](double x) { return x / total; });

    // Select an action based on the probabilities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return dist(gen);
}
