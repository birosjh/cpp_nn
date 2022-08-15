#include <iostream>
#include <Eigen/Dense>
#include "Network/network.h"

using Eigen::MatrixXd;
 
int main()
{

  vector<int> sizes{ 10, 20, 30 }; 

  Network nn(sizes);

  std::cout << nn << std::endl;
}