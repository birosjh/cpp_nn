#include "base_data_loader.h"
#include <Eigen/Dense>
#include <vector>
#include "mnist/mnist_reader.hpp"

using Eigen::MatrixXf;

class FashionMNISTLoader
{
private:
    int m_batch_size;
    int m_train_num_batches;
    int m_test_num_batches;

    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> m_dataset;
    std::vector<int> training_indices;
    std::vector<int> test_indices;

public:
    FashionMNISTLoader(int batch_size, bool shuffle_train = true);
    Batch nextBatch();
    int num_training_batches();
    int num_test_batches();
};