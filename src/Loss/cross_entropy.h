#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include <eigen/Dense>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;

float cross_entropy(MatrixXf predictions, std::vector<int> ground_truth);

MatrixXf cross_entropy_prime(MatrixXf probabilities, std::vector<int> ground_truth);

#endif // CROSS_ENTROPY_H

