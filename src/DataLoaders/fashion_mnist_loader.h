#ifndef LOADER_H
#define LOADER_H

#include "base_data_loader.h"
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include "mnist/mnist_reader.hpp"

using Eigen::MatrixXf;

class FashionMNISTLoader
{
private:
    int m_batch_size;
    int m_train_num_batches;
    int m_test_num_batches;
    bool m_shuffle_train;

    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> m_dataset;
    std::vector<int> m_training_indices;
    std::vector<int> m_test_indices;

    void setTrainIndices();
    void setTestIndices();
    Batch createBatch(std::vector<std::vector<float>> images, std::vector<int> labels);
    std::vector<Batch> loadBatches(std::vector<std::vector<float>> image_set, std::vector<uint8_t> label_set, std::vector<int> indices, int number_of_images);

public:
    FashionMNISTLoader(int batch_size, bool shuffle_train = true);
    std::vector<Batch> loadTrainingBatches();
    std::vector<Batch> loadTestBatches();
    int num_training_batches();
    int num_test_batches();
};

#endif // LOADER_H