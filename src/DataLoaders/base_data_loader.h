#include <string>
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXf;

struct Batch
{
    MatrixXf images;
    std::vector<int> labels;
};

class BaseDataLoader
{
public:
    virtual Batch nextBatch();
};