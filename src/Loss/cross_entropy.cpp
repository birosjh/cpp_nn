#include "cross_entropy.h"

#include <cmath>
#include <vector>
#include <eigen/Dense>
#include <iostream>

float cross_entropy(MatrixXf predictions, std::vector<int> ground_truth) {

    MatrixXf truth_onehot = MatrixXf::Zero(predictions.cols(), predictions.rows());

    for (int idx = 0; idx < predictions.rows(); ++idx) {
        truth_onehot.row(idx)[ground_truth[idx]] = 1;
    }

    auto element_wise_log = [](float value) { return std::log(value); };

    auto resultant = truth_onehot * (-1 * predictions.unaryExpr(element_wise_log));

    return resultant.sum();
};

MatrixXf cross_entropy_prime(MatrixXf probabilities, std::vector<int> ground_truth) {

    MatrixXf truth_onehot = MatrixXf::Zero(probabilities.rows(), probabilities.cols());

    for (int idx = 0; idx < probabilities.cols(); ++idx) {
        truth_onehot.col(idx)[ground_truth[idx]] = 1;
    }

    return probabilities - truth_onehot;
}