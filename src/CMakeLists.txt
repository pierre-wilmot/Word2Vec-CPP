add_executable(Word2Vec main.cpp)

add_executable(Distance distance.c)
# Link the standard math library (used for sqrt) 
target_link_libraries(Distance m) 
