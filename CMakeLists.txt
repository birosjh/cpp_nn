cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpp_nn)

set(CMAKE_CXX_STANDARD 17)

# add the executable
add_executable(cpp_nn src/main.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_subdirectory(external/eigen3)
include_directories(PRIVATE external/eigen3)

add_subdirectory(src)

target_include_directories(${PROJECT_NAME}
                            PUBLIC network
                            PUBLIC dataloaders
                            PUBLIC utils
                            PUBLIC loss
                            PUBLIC accuracy)

target_link_directories(${PROJECT_NAME} 
                        PRIVATE network
                        PRIVATE dataloaders
                        PRIVATE utils
                        PRIVATE loss
                        PRIVATE accuracy)

target_link_libraries(${PROJECT_NAME} 
                        network 
                        dataloaders
                        utils
                        loss
                        accuracy)