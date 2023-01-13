#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

class Network {

    private:
        int m_numLayers;
        float m_learning_rate;
        vector<int> m_sizes;
        vector<MatrixXf> m_weights;
        vector<VectorXf> m_biases;
        vector<MatrixXf> m_layer_outputs;
        vector<MatrixXf> m_layer_activations;

        void initializeWeightsAndBiases();

    public:
        Network(vector<int>, float learning_rate);

        MatrixXf forward(MatrixXf input);

        vector<MatrixXf> getLayers();

        void backward(MatrixXf probabilities, std::vector<int> ground_truth);

};

std::ostream& operator<<(std::ostream& os, Network& network);

#endif // NETWORK_H