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
        vector<int> m_sizes;
        vector<MatrixXf> m_weights;
        vector<VectorXf> m_biases;

        void initializeWeightsAndBiases();

    public:
        Network(vector<int>);

        MatrixXf forward(MatrixXf input);

        vector<MatrixXf> getLayers();

        void backward(float loss);

};

std::ostream& operator<<(std::ostream& os, Network& network);

#endif // NETWORK_H