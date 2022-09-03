#include <cmath>

float sigmoid(const float& input) {
    return 1.0 / (1.0 + exp(-input));
};