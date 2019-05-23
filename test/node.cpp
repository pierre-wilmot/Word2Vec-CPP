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

#include "node.hpp"
#include "tensor.hpp"
#include "placeholder.hpp"
#include <gtest/gtest.h>

using ArrayType = fetch::math::Tensor<int, 2>;

TEST(node_test, node_placeholder)
{
  fetch::ml::Node<ArrayType, fetch::ml::ops::PlaceHolder<ArrayType, 2>>
                           placeholder("PlaceHolder");
  ArrayType data({5, 5});
  placeholder.SetData(data);

  EXPECT_EQ((placeholder.fetch::ml::template Ops<ArrayType, 2>::Forward({}).Storage()), data.Storage());
  EXPECT_EQ(placeholder.Evaluate().Storage(), data.Storage());
}
