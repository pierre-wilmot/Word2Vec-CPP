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

#include "embeddings.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

template <typename T>
class EmbeddingsTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<int, float, double>;
TYPED_TEST_CASE(EmbeddingsTest, MyTypes);

TYPED_TEST(EmbeddingsTest, forward_shape)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  using SizeType = typename ArrayType::SizeType;
  fetch::ml::ops::Embeddings<ArrayType> e(SizeType(100), SizeType(60));
  ArrayType                             input({1, 10});
  for (unsigned int i(0); i < 10; ++i)
  {
    input.Set(0, i, TypeParam(i));
  }
  ArrayType output = e.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(input)});

  std::array<SizeType, 2> expected_output_shape({10, 60});
  ASSERT_EQ(output.shape(), expected_output_shape);
}

TYPED_TEST(EmbeddingsTest, forward)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  using SizeType = typename ArrayType::SizeType;
  fetch::ml::ops::Embeddings<ArrayType> e(SizeType(10), SizeType(6));

  ArrayType weights({10, 6});
  for (unsigned int i(0); i < 10; ++i)
  {
    for (unsigned int j(0); j < 6; ++j)
    {
      weights.Set(i, j, TypeParam(i * 10 + j));
    }
  }

  e.SetData(weights);
  ArrayType input({1, 2});
  input.Set(0, 0, TypeParam(3));
  input.Set(0, 1, TypeParam(5));
  ArrayType output = e.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(input)});

  std::array<SizeType, 2> expected_output_shape({2, 6});
  ASSERT_EQ(output.shape(), expected_output_shape);

  std::vector<int> row1_gt({30, 31, 32, 33, 34, 35});
  std::vector<int> row2_gt({50, 51, 52, 53, 54, 55});
  for (unsigned int i(0); i < 6; ++i)
    {
      EXPECT_EQ(output.Get(0, i), TypeParam(row1_gt[i]));
      EXPECT_EQ(output.Get(1, i), TypeParam(row2_gt[i]));
    }
}

TYPED_TEST(EmbeddingsTest, backward)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  using SizeType = typename ArrayType::SizeType;
  fetch::ml::ops::Embeddings<ArrayType> e(SizeType(10), SizeType(6));

  ArrayType weights({10, 6});
  for (unsigned int i(0); i < 10; ++i)
  {
    for (unsigned int j(0); j < 6; ++j)
    {
      weights.Set(i, j, TypeParam(i * 10 + j));
    }
  }

  e.SetData(weights);
  ArrayType input({1, 2});
  input.Set(0, 0, TypeParam(3));
  input.Set(0, 1, TypeParam(5));
  ArrayType output = e.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(input)});

  ArrayType errorSignal({2, 6});
  for (unsigned int j(0); j < 6; ++j)
  {
    errorSignal.Set(0, j, TypeParam(j + 0));
    errorSignal.Set(1, j, TypeParam(j + 6));
  }
  e.fetch::ml::template Ops<ArrayType, 2>::Backward({input}, errorSignal);
  e.Step(TypeParam(1));  

  output = e.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(input)});
  std::vector<int> row1_gt({30, 32, 34, 36, 38, 40});
  std::vector<int> row2_gt({56, 58, 60, 62, 64, 66});
  for (unsigned int j(0); j < 6; ++j)
    {
      EXPECT_EQ(output.Get(0, j), TypeParam(row1_gt[j]));
      EXPECT_EQ(output.Get(1, j), TypeParam(row2_gt[j]));
    }

  // Taking a second step, this should have no effect as the grdient accumulation buffer is expected to be cleared
  e.Step(TypeParam(1));  
  output = e.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(input)});
  for (unsigned int j(0); j < 6; ++j)
    {
      EXPECT_EQ(output.Get(0, j), TypeParam(row1_gt[j]));
      EXPECT_EQ(output.Get(1, j), TypeParam(row2_gt[j]));
    }  
}
