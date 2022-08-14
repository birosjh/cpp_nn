#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using std::vector;

class Network {

    private:
        int m_numLayers;
        vector<int> m_sizes;
        vector<MatrixXd> m_weights;
        vector<MatrixXd> m_biases;

        void initializeWeightsAndBiases();

    public:
        Network(vector<int>);

        MatrixXd forward(MatrixXd input);

        void backward(float loss);

};

#endif // NETWORK_H