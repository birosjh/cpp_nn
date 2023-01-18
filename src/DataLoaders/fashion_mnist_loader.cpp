#include "fashion_mnist_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <numeric>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;

FashionMNISTLoader::FashionMNISTLoader(int batch_size, bool shuffle_train):
    m_batch_size(batch_size),
    m_shuffle_train(shuffle_train)
{

    m_dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("external/fashion_mnist/data/fashion");

    auto training_size = m_dataset.training_images.size();
    auto testing_size = m_dataset.test_images.size();

    // Create training and test indices
    m_training_indices.resize(training_size);
    m_test_indices.resize(testing_size);

    // Set Batch Size
    m_train_num_batches = (training_size - 1) / m_batch_size;
    m_test_num_batches = (testing_size - 1) / m_batch_size;

    std::cout << "Training Batches: " << m_dataset.training_images.size() << std::endl;
    std::cout << "Test Batches: " << m_dataset.test_images.size() << std::endl;
};

void FashionMNISTLoader::setTrainIndices() {

    std::iota(std::begin(m_training_indices), std::end(m_training_indices), 0);

    // Shuffle training indices
    if (m_shuffle_train) {
        auto engine = std::default_random_engine {};
        std::shuffle(std::begin(m_training_indices), std::end(m_training_indices), engine);
    }

}

void FashionMNISTLoader::setTestIndices() {

    std::iota(std::begin(m_test_indices), std::end(m_test_indices), 0);

}

std::vector<Batch> FashionMNISTLoader::loadTrainingBatches() {

    setTrainIndices();

    auto number_of_images = m_batch_size * m_train_num_batches;

    auto batches = loadBatches(m_dataset.training_images, m_dataset.training_labels, m_training_indices, number_of_images);

    return batches;

};

std::vector<Batch> FashionMNISTLoader::loadTestBatches() {

    setTestIndices();

    auto number_of_images = m_batch_size * m_test_num_batches;

    auto batches = loadBatches(m_dataset.test_images, m_dataset.test_labels, m_test_indices, number_of_images);

    return batches;

};

std::vector<Batch> FashionMNISTLoader::loadBatches(std::vector<std::vector<float>> image_set, std::vector<uint8_t> label_set, std::vector<int> indices, int number_of_images) {

    
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    std::vector<Batch> batches;

    int row_index = 0;
    
    for (int image_index = 0; image_index < number_of_images; image_index++) {

        images.push_back(image_set[indices[image_index]]);
        labels.push_back((int)label_set[indices[image_index]]);

        row_index++;

        if (row_index == m_batch_size) {
            Batch batch = createBatch(images, labels);

            batches.push_back(batch);

            row_index = 0;
            images.clear();
            labels.clear();
        }
        
    }

    return batches;

}

Batch FashionMNISTLoader::createBatch(std::vector<std::vector<float>> images, std::vector<int> labels) {

    MatrixXf matrix_batch(m_batch_size, 784);

    for (int row_index = 0; row_index < images.size(); row_index++) {
        matrix_batch.row(row_index) = VectorXf::Map(&images[row_index][0], images[row_index].size());
    }

    Batch batch;

    batch.images = matrix_batch;
    batch.labels = labels;

    return batch;

}

int FashionMNISTLoader::num_training_batches() {
    return m_train_num_batches;
};

int FashionMNISTLoader::num_test_batches() {
    return m_test_num_batches;
};