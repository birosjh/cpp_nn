
add_library(network Network/network.cpp)

target_include_directories(network
                            PUBLIC ${PROJECT_SOURCE_DIR}/external/eigen3)