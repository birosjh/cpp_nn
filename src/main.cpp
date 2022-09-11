#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Network/network.h"
#include "DataLoaders/fashion_mnist_loader.h"
#include "Utils/utils.h"
#include "Loss/cross_entropy.h"
#include "Metrics/accuracy.h"

using Eigen::MatrixXf;
 
int main()
{

  FashionMNISTLoader dataloader = FashionMNISTLoader(16);
  
  // Define Network
  std::vector<int> sizes{ 784, 42, 10 };
  Network nn(sizes);

  // Print Network Layout
  std::cout << nn << std::endl;

  std::cout << "Num batches: " << dataloader.num_training_batches() << std::endl;

  // Run a Batch
  auto batch = dataloader.nextBatch();

  MatrixXf output = nn.forward(batch.images);

  // std::cout << predictions << std::endl;

  float loss = cross_entropy(output, batch.labels);

  std::cout << "Loss: " << loss << std::endl;

  // Get max of each
  std::vector<int> predictions = argmax(output);

  for (auto value : predictions) { 
    std::cout << value << std::endl;
  }

  // Calculate Accuracy
  auto acc = accuracy(predictions, batch.labels);

  std::cout << "Accuracy: " << acc << std::endl;
}