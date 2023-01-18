#include "Network/network.h"
#include "DataLoaders/fashion_mnist_loader.h"
#include "Trainer/trainer.h"

#include <iostream>
#include <vector>
 
int main()
{
  int epochs = 5;

  FashionMNISTLoader dataloader = FashionMNISTLoader(16);
  
  // Define Network
  std::vector<int> sizes{ 784, 42, 10 };
  Network nn(sizes, 0.2);

  // Print Network Layout
  std::cout << nn << std::endl;
  
  Trainer trainer(nn, dataloader, epochs);

  trainer.fit();

  // trainer.test();
}