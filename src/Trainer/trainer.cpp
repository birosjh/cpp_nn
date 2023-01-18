#include "trainer.h"
#include "network.h"
#include "fashion_mnist_loader.h"
#include "utils.h"
#include "cross_entropy.h"
#include "accuracy.h"
#include <vector>


Trainer::Trainer(Network &network, FashionMNISTLoader &dataloader, int epochs):
    m_network(network),
    m_dataloader(dataloader),
    m_epochs(epochs)
{};

void Trainer::fit() {

    // Run a Batch
    for (int epoch = 0; epoch < m_epochs; epoch++) {

        int batch_count = m_dataloader.num_training_batches();

        std::cout << "Epoch: " << epoch << std::endl;

        int epoch_loss = 0;
        int epoch_acc = 0;
        
        std::cout << "Total batches: " << batch_count << std::endl;

        std::vector<Batch> batches = m_dataloader.loadTrainingBatches();

        for (auto batch : batches) {

            MatrixXf probabilities = m_network.forward(batch.images);

            float loss = cross_entropy(probabilities, batch.labels);

            std::vector<int> predictions = argmax(probabilities);
            auto acc = accuracy(predictions, batch.labels);

            m_network.backward(probabilities, batch.labels);

            epoch_loss += loss;
            epoch_acc += acc;

        }

        std::cout << "Loss: " << epoch_loss / batch_count << std::endl;
        std::cout << "Accuracy: " << epoch_acc / batch_count << std::endl;
    }

};

void Trainer::test() {

    int batch_count = m_dataloader.num_test_batches();

    // Run a Batch
    for (int epoch = 0; epoch < m_epochs; epoch++) {

        std::cout << "Epoch: " << epoch << std::endl;

        int epoch_loss = 0;
        int epoch_acc = 0;

        std::vector<Batch> batches = m_dataloader.loadTestBatches();

        for (auto batch : batches) {

            MatrixXf probabilities = m_network.forward(batch.images);

            std::vector<int> predictions = argmax(probabilities);
            auto acc = accuracy(predictions, batch.labels);

            std::cout << "Accuracy: " << acc << std::endl;

        }

        std::cout << "Loss: " << epoch_loss / batch_count << std::endl;
        std::cout << "Accuracy: " << epoch_acc / batch_count << std::endl;

    }

};