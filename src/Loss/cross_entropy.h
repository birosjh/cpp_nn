#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include <eigen/Dense>
#include <vector>

using Eigen::MatrixXf;

float cross_entropy(MatrixXf predictions, std::vector<int> ground_truth);

#endif // CROSS_ENTROPY_H

