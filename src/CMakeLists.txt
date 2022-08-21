set(NETWORK network)
add_library(${NETWORK} Network/network.cpp)
add_library(activations Activations/activations.cpp)

target_include_directories(activations
                            PUBLIC ${PROJECT_SOURCE_DIR}/external/eigen3)

target_include_directories(network
                            PUBLIC ${PROJECT_SOURCE_DIR}/external/eigen3
                            PUBLIC activations)

target_link_directories(${NETWORK} PRIVATE activations)

target_link_libraries(${NETWORK} activations)

