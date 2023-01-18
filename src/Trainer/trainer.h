#ifndef TRAINER_H
#define TRAINER_H

#include "fashion_mnist_loader.h"
#include "network.h"

class Trainer {

    private:
        FashionMNISTLoader m_dataloader;
        Network m_network;
        int m_epochs;

    public:
        Trainer(Network &network, FashionMNISTLoader &dataloader, int epochs);

        void fit();

        void test();

};

#endif // TRAINER_H