set(NETWORK network)
add_library(${NETWORK} Network/network.cpp)
add_library(activations Activations/activations.cpp)
add_library(dataloaders DataLoaders/fashion_mnist_loader.cpp DataLoaders/base_data_loader.h)
add_library(utils Utils/utils.cpp)
add_library(loss Loss/cross_entropy.cpp)
add_library(accuracy Metrics/accuracy.cpp)

target_include_directories(dataloaders
                            PUBLIC ${PROJECT_SOURCE_DIR}/external/fashion_mnist/include)

target_link_directories(dataloaders PRIVATE ${PROJECT_SOURCE_DIR}/external/fashion_mnist/include)

target_include_directories(${NETWORK}
                            PUBLIC activations)

target_link_directories(${NETWORK} PRIVATE activations)

target_link_libraries(${NETWORK} activations)