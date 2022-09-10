#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXf;

std::vector<int> argmax(MatrixXf input);

#endif // ACTIVATIONS_H