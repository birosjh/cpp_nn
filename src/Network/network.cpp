#include "network.h"
#include "activations.h"
#include "cross_entropy.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

Network::Network(vector<int> sizes, float learning_rate) {

    m_sizes = sizes;

    m_learning_rate = learning_rate;
    
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

    MatrixXf output = input.transpose();

    for (int idx = 0; idx < m_weights.size(); idx++) {
        output = m_weights[idx] * output;
        output.colwise() += m_biases[idx];
        m_layer_outputs.push_back(output);

        output = sigmoid(output);
        m_layer_activations.push_back(output);
    }

    output = softmax(output);

    return output;
};

void Network::backward(MatrixXf probabilities, std::vector<int> ground_truth) {

    vector<MatrixXf> weight_gradients;
    vector<VectorXf> bias_gradients;

    // Initialize Gradient Matrices
    for (int idx = 0; idx < m_weights.size(); idx++) {
        MatrixXf weight_gradient_layer = m_weights[idx];
        weight_gradient_layer.setZero();
        weight_gradients.push_back(weight_gradient_layer);

        VectorXf bias_gradient_layer = m_biases[idx];
        bias_gradient_layer.setZero();
        bias_gradients.push_back(bias_gradient_layer);
    }

    // Collect Gradients
    MatrixXf delta = cross_entropy_prime(probabilities, ground_truth);    

    bias_gradients.back() += delta.rowwise().sum();
    weight_gradients.back() += delta * m_layer_activations.end()[-2].transpose();

    for (int idx = m_weights.size() - 2; idx > 0; idx--) {

        auto layer_output = m_layer_outputs[idx];
        auto diff_weighted_input = sigmoid_prime(layer_output);

        VectorXf delta = (m_weights[idx].transpose() * delta) * diff_weighted_input;

        bias_gradients.back() = delta;
        weight_gradients.back() = delta * m_layer_activations[idx - 1].transpose();
    }

    // Apply Gradient Descent
    for (int idx = 0; idx < weight_gradients.size(); idx++) {

        m_weights[idx] = m_weights[idx] - m_learning_rate * weight_gradients[idx];
        m_biases[idx] = m_biases[idx] - m_learning_rate * bias_gradients[idx];

    }

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