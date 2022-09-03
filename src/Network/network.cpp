#include "network.h"
#include "activations.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

Network::Network(vector<int> sizes) {

    m_sizes = sizes;
    
    Network::initializeWeightsAndBiases();

};

void Network::initializeWeightsAndBiases() {
    m_weights.reserve(m_sizes.size());
    m_biases.reserve(m_sizes.size());

    for (int idx = 1; idx < m_sizes.size(); idx++) {
        m_weights.push_back(
            MatrixXf::Random(m_sizes[idx], m_sizes[idx - 1])
        );
        m_biases.push_back(
            VectorXf::Random(m_sizes[idx])
        );
    }
};

MatrixXf Network::forward(MatrixXf input) {

    MatrixXf output = m_weights[0] * input.transpose();
    output.colwise() += m_biases[0];

    output = output.unaryExpr(std::ref(sigmoid));

    for (int idx = 1; idx < m_weights.size(); idx++) {
        output = m_weights[idx] * output;
        output.colwise() += m_biases[idx];
        output = output.unaryExpr(std::ref(sigmoid));
    }

    return output;

};

vector<MatrixXf> Network::getLayers() {

    return m_weights;

};

std::ostream& operator<<(std::ostream& out, Network& network){
    vector<MatrixXf> layers = network.getLayers();

    for (int idx = 0; idx <= layers.size() - 1; ++idx)
    {   
        out << "Layer: " << idx << std::endl;
        out << layers[idx].rows() << "x" << layers[idx].cols() << std::endl;
    }

    return out;
};