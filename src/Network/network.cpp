#include "network.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Vector3d;

Network::Network(vector<int> sizes) {

    m_sizes = sizes;
    
    Network::initializeWeightsAndBiases();

};

void Network::initializeWeightsAndBiases() {
    m_weights.reserve(m_sizes.size());
    m_biases.reserve(m_sizes.size());

    for (int idx = 1; idx < m_sizes.size(); idx++) {
        m_weights.push_back(
            MatrixXd::Random(m_sizes[idx - 1], m_sizes[idx])
        );
        m_biases.push_back(
            MatrixXd::Random(m_sizes[idx - 1], 1)
        );
    }
};

MatrixXd Network::forward(MatrixXd input) {

    MatrixXd output = input * m_weights[0] + m_biases[0];

    for (int idx = 1; idx < m_sizes.size(); idx++) {
        output = output * m_weights[idx] + m_biases[idx];
    }

    return output;

};