#include "accuracy.h"

float accuracy(std::vector<int> predictions, std::vector<int> labels) {

    float num_correct = 0.0;

    for (int idx = 0; idx < predictions.size(); ++idx) {
        if (predictions[idx] == labels[idx]) {
            num_correct++;
        }
    }

    return num_correct / predictions.size();
}