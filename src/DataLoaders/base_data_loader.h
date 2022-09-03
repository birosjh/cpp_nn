#include <string>
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXf;

class BaseDataLoader
{
public:
    virtual MatrixXf nextBatch();
};