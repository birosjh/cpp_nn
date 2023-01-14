#include "utils.h"
#include <iostream>
#include <vector>
#include <eigen/Dense>

using Eigen::MatrixXf;

std::vector<int> argmax(MatrixXf input) {

    std::vector<int> max_indices(input.cols());
    std::vector<int> max_values;

    for (int idx = 0; idx < input.cols(); ++idx) {

        max_values.push_back(input.col(idx).maxCoeff(&max_indices[idx]));

    }

    return max_indices;
}

std::vector<float> max(MatrixXf input) {

    std::vector<float> max_values;

    for (int idx = 0; idx < input.cols(); ++idx) {

        max_values.push_back(input.col(idx).maxCoeff());

    }

    return max_values;
}

MatrixXf hadamard(MatrixXf &matrix_a, MatrixXf &matrix_b) {

    MatrixXf product = matrix_a;
    product.setZero();

    for(int row_id = 0; row_id < matrix_a.rows(); row_id++) {
        product.row(row_id) += matrix_a.row(row_id) * matrix_b.row(row_id);
    }

    return product;
}