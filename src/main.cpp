#include <iostream>
#include <Eigen/Dense>
#include "Network/network.h"

using Eigen::MatrixXd;
 
int main()
{

  vector<int> sizes{ 10, 20, 30 }; 

  Network nn(sizes);

  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);

  std::cout << m << std::endl;
}