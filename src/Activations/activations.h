

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

MatrixXf sigmoid(MatrixXf input);
MatrixXf sigmoid_prime(MatrixXf input);

MatrixXf softmax(MatrixXf input);

#endif // ACTIVATIONS_H