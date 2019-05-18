if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


project(KrakenCache)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 14)

add_executable(Word2Vec main.cpp)

add_executable(Distance distance.c)
# Link the standard math library (used for sqrt) 
target_link_libraries(Distance m) 
