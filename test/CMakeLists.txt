hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

add_executable(TensorTest tensor_indexing.cpp)
target_link_libraries(TensorTest PUBLIC GTest::main)
add_test(TensorTest, TensorTest)

add_executable(EmbeddingsTest embeddings.cpp)
target_link_libraries(EmbeddingsTest PUBLIC GTest::main)
add_test(EmbeddingsTest, EmbeddingsTest)

add_executable(GraphTest graph.cpp)
target_link_libraries(GraphTest PUBLIC GTest::main)
add_test(GraphTest, GraphTest)

add_executable(MatrixMultiplyTest matrix_multiply.cpp)
target_link_libraries(MatrixMultiplyTest PUBLIC GTest::main)
add_test(MatrixMultiplyTest, MatrixMultiplyTest)

add_executable(NodeTest node.cpp)
target_link_libraries(NodeTest PUBLIC GTest::main)
add_test(NodeTest, NodeTest)

add_executable(PlaceholderTest placeholder.cpp)
target_link_libraries(PlaceholderTest PUBLIC GTest::main)
add_test(PlaceholderTest, PlaceholderTest)

add_executable(SigmoidTest sigmoid.cpp)
target_link_libraries(SigmoidTest PUBLIC GTest::main)
add_test(SigmoidTest, SigmoidTest)

add_executable(WeightsTest weights.cpp)
target_link_libraries(WeightsTest PUBLIC GTest::main)
add_test(WeightsTest, WeightsTest)
