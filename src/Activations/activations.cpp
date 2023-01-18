#include <cmath>
#include <eigen/Dense>
#include <iostream>

using Eigen::MatrixXf;

float sigmoid_base(const float& value) {
    return 1.0 / (1.0 + exp(-value));
}

float sigmoid_prime_base(const float& value) {
    return sigmoid_base(value) * (1 - sigmoid_base(value));
}

MatrixXf sigmoid(MatrixXf input) {
    
    MatrixXf output = input.unaryExpr(
        std::ref(sigmoid_base)
    );

    return output;
}

MatrixXf sigmoid_prime(MatrixXf input) {
    
    MatrixXf output = input.unaryExpr(
        std::ref(sigmoid_prime_base)
    );

    return output;
}

MatrixXf softmax(MatrixXf input) {

    MatrixXf softmaxed = MatrixXf::Zero(input.rows(), input.cols());

    for(int idx = 0; idx < input.cols(); ++idx) {
        auto powered = input.col(idx).array().exp();
        float powered_sum = powered.sum();

        softmaxed.col(idx) = powered;
        softmaxed.col(idx) /= powered_sum;
    }

    return softmaxed;
}