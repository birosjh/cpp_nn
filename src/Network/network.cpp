#include "network.h"
#include "activations.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
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
            MatrixXd::Random(m_sizes[idx], m_sizes[idx - 1])
        );
        m_biases.push_back(
            MatrixXd::Random(m_sizes[idx], 1)
        );
    }
};

MatrixXd Network::forward(MatrixXd input) {

    VectorXd flattened = Network::flatten(input);

    VectorXd output = m_weights[0] * flattened + m_biases[0];
    output = output.unaryExpr(std::ref(sigmoid));

    for (int idx = 1; idx < m_weights.size(); idx++) {
        output = m_weights[idx] * output + m_biases[idx];
        output = output.unaryExpr(std::ref(sigmoid));
    }

    return output;

};

vector<MatrixXd> Network::getLayers() {

    return m_weights;

};

VectorXd Network::flatten(MatrixXd original) {

    Eigen::Map<VectorXd> flattened(original.data(), original.size());

    return flattened;

}

std::ostream& operator<<(std::ostream& out, Network& network){
    vector<MatrixXd> layers = network.getLayers();

    for (int idx = 0; idx <= layers.size() - 1; ++idx)
    {   
        out << "Layer: " << idx << std::endl;
        out << layers[idx].rows() << "x" << layers[idx].cols() << std::endl;
    }

    return out;
};