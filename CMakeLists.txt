### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.19.87.tar.gz"
    SHA1 "1b0041a06357d2c9f1f1a7e93ff3132c36411719"
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(Word2Vec)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 14)
enable_testing()

include_directories(include)
add_subdirectory(src)
add_subdirectory(test)
