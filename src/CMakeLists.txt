add_library(network Network/network.cpp)
add_library(activations Activations/activations.cpp)
add_library(dataloaders DataLoaders/fashion_mnist_loader.cpp DataLoaders/base_data_loader.h)
add_library(utils Utils/utils.cpp)
add_library(loss Loss/cross_entropy.cpp)
add_library(accuracy Accuracy/accuracy.cpp)
add_library(trainer Trainer/trainer.cpp)

target_include_directories(dataloaders
                            PUBLIC ${PROJECT_SOURCE_DIR}/external/fashion_mnist/include)

target_link_directories(dataloaders PRIVATE ${PROJECT_SOURCE_DIR}/external/fashion_mnist/include)

target_include_directories(network
                            PUBLIC activations
                            PUBLIC loss
                            PUBLIC utils)

target_link_directories(network
                          PRIVATE activations
                          PRIVATE loss
                          PRIVATE utils)

target_link_libraries(network activations loss utils)

target_include_directories(trainer
                            PUBLIC network
                            PUBLIC utils
                            PUBLIC dataloaders
                            PUBLIC loss
                            PUBLIC accuracy)

target_link_directories(trainer
                          PRIVATE network
                          PRIVATE utils
                          PRIVATE dataloaders
                          PRIVATE loss
                          PRIVATE accuracy)

target_link_libraries(trainer network utils dataloaders loss accuracy)

