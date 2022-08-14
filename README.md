# CPP NN

This is a basic implementation of a neural network written in C++.  It is based on the code from the Neural Networks and Deep Learning book (http://neuralnetworksanddeeplearning.com)


### Docker Commands

To build the image:

```bash
docker build . -t cpp_nn
```

To start the image:

```bash
docker run -itd --name cpp_nn -v $(pwd):/app cpp_nn
```

To enter the docker container:

```bash
docker exec -it cpp_nn bash
```

### Building the Project

These commands must be run from inside of the docker container.

To run cmake:

```bash
./configure.sh
```

To build:

```bash
./build.sh
```

To run the executable:

```bash
./run.sh
```
