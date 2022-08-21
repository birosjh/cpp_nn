#include <Eigen/Dense>
#include <cmath>

double sigmoid(const double& input) {
    return 1.0 / (1.0 + exp(-input));
};