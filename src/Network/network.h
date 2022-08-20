#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

class Network {

    private:
        int m_numLayers;
        vector<int> m_sizes;
        vector<MatrixXd> m_weights;
        vector<MatrixXd> m_biases;

        void initializeWeightsAndBiases();
        VectorXd flatten(MatrixXd original);

    public:
        Network(vector<int>);

        MatrixXd forward(MatrixXd input);

        vector<MatrixXd> getLayers();

        void backward(float loss);

};

std::ostream& operator<<(std::ostream& os, Network& network);

#endif // NETWORK_H