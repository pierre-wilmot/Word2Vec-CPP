//------------------------------------------------------------------------------
//
//   Copyright 2018-2019 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "graph.hpp"
#include "tensor.hpp"
#include "placeholder.hpp"

#include <gtest/gtest.h>

using ArrayType = typename fetch::math::Tensor<int, 2>;

TEST(graph_test, node_placeholder)
{
  fetch::ml::Graph<ArrayType> g;
  g.AddNode<fetch::ml::ops::PlaceHolder<ArrayType, 2>>("Input", {});

  ArrayType data({1, 8});
  std::vector<int> values({1, 2, 3, 4, 5, 6, 7, 8});
  for (uint64_t i(0) ; i < data.Size() ; ++i)
  {
    data.Set(0, i, values[i]);
  }

  g.SetInput("Input", data);
  ArrayType prediction = g.Evaluate("Input");

  // test correct values
  for (uint64_t i(0) ; i < prediction.Size() ; ++i)
  {
    EXPECT_EQ(prediction.Get(0, i), values[i]);
  }  
}

TEST(graph_test, getStateDict)
{
  fetch::ml::Graph<fetch::math::Tensor<float, 2>>     g;
  fetch::ml::StateDict<fetch::math::Tensor<float, 2>> sd = g.StateDict();

  EXPECT_TRUE(sd.weights_ == nullptr);
  EXPECT_TRUE(sd.dict_.empty());
}

TEST(graph_test, no_such_node_test)  // Use the class as a Node
{
  fetch::ml::Graph<ArrayType> g;
  g.template AddNode<fetch::ml::ops::PlaceHolder<ArrayType, 2>>("Input", {});
  ArrayType data({5, 10});
  g.SetInput("Input", data);
  ASSERT_ANY_THROW(g.Evaluate("FullyConnected"));
}
