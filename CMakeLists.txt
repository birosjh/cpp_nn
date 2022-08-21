cmake_minimum_required(VERSION 3.10)

# set the project name
project(cpp_nn)

set(CMAKE_CXX_STANDARD 17)

# add the executable
add_executable(cpp_nn src/main.cpp)

add_subdirectory(src)
add_subdirectory(external/eigen3)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_include_directories(${PROJECT_NAME}
                            PRIVATE external/eigen3
                            PUBLIC network)

target_link_directories(${PROJECT_NAME} PRIVATE network)

target_link_libraries(${PROJECT_NAME} network)