#include "fashion_mnist_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <numeric>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;

FashionMNISTLoader::FashionMNISTLoader(int batch_size, bool shuffle_train) {
    m_batch_size = batch_size;
    m_dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("external/fashion_mnist/data/fashion");

    // Create training and test indices
    training_indices.resize(m_dataset.training_images.size());
    std::iota(std::begin(training_indices), std::end(training_indices), 0);

    test_indices.resize(m_dataset.test_images.size());
    std::iota(std::begin(test_indices), std::end(test_indices), 0);

    // Set Batch Size
    m_train_num_batches = training_indices.size() / m_batch_size;
    m_test_num_batches = test_indices.size() / m_batch_size;

    std::cout << "Training Size: " << m_dataset.training_images.size() << std::endl;
    std::cout << "Test Size: " << m_dataset.test_images.size() << std::endl;

    // Shuffle training indices
    if (shuffle_train) {
        auto engine = std::default_random_engine {};
        std::shuffle(std::begin(training_indices), std::end(training_indices), engine);
    }
}

MatrixXf FashionMNISTLoader::nextBatch() {

    MatrixXf matrix_batch(m_batch_size, 784);
    VectorXf row;
    std::vector<float> image;
    
    int idx = 0;

    while (idx < m_batch_size && training_indices.size() > m_batch_size) {

        image = m_dataset.training_images[training_indices.back()];
        
        matrix_batch.row(idx) = Eigen::VectorXf::Map(&image[0], image.size());;
        
        training_indices.pop_back();

        idx++;
    }

    return matrix_batch;
}

int FashionMNISTLoader::num_training_batches() {
    return m_train_num_batches;
};

int FashionMNISTLoader::num_test_batches() {
    return m_test_num_batches;
};