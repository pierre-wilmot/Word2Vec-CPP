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

#include "matrix_multiply.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

template <typename T>
class MatrixMultiplyTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<fetch::math::Tensor<int, 2>, fetch::math::Tensor<float, 2>,
                                 fetch::math::Tensor<double, 2>>;
TYPED_TEST_CASE(MatrixMultiplyTest, MyTypes);

TYPED_TEST(MatrixMultiplyTest, forward_test)
{
  TypeParam a({1, 5});
  TypeParam b({5, 4});

  std::vector<int> data({1, 2, -3, 4, 5});
  std::vector<int> weights({-11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44, 51, 52, 53, 54});
  std::vector<int> results({357, 388, 397, 406});

  for (std::uint64_t i(0); i < data.size(); ++i)
  {
    a.Set(0, i, typename TypeParam::Type(data[i]));
  }
  for (std::uint64_t i(0); i < 5; ++i)
  {
    for (std::uint64_t j(0); j < 4; ++j)
    {
      b.Set(i, j, typename TypeParam::Type(weights[i * 4 + j]));
    }
  }

  fetch::ml::ops::MatrixMultiply<TypeParam> op;
  TypeParam                                 prediction = op.fetch::ml::template Ops<TypeParam, 2>::Forward({std::cref(a), std::cref(b)});

  // test correct values
  std::array<typename TypeParam::SizeType, 2> expected_output_shape({1, 4});
  ASSERT_EQ(prediction.shape(), expected_output_shape);
  for (std::size_t i(0) ; i < results.size() ; ++i)
    {
      ASSERT_EQ(prediction.Get(0, i), results[i]);
    }
}

TYPED_TEST(MatrixMultiplyTest, backward_test)
{
  TypeParam a({1, 5});
  TypeParam b({5, 4});
  TypeParam error({1, 4});

  std::vector<int> data({1, 2, -3, 4, 5});
  std::vector<int> weights({-11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44, 51, 52, 53, 54});
  std::vector<int> errorSignal({1, 2, 3, -4});
  std::vector<int> inputGrad({-4, 38, 58, 78, 98});
  std::vector<int> weightsGrad({1, 2, 3, -4, 2, 4, 6, -8, -3, -6, -9, 12, 4, 8, 12, -16, 5, 10, 15, -20});

  for (std::uint64_t i(0); i < data.size(); ++i)
  {
    a.Set(0, i, typename TypeParam::Type(data[i]));
  }
  for (std::uint64_t i(0); i < 5; ++i)
  {
    for (std::uint64_t j(0); j < 4; ++j)
    {
      b.Set(i, j, typename TypeParam::Type(weights[i * 4 + j]));
    }
  }
  for (std::uint64_t i(0); i < errorSignal.size(); ++i)
  {
    error.Set(0, i, typename TypeParam::Type(errorSignal[i]));
  }

  fetch::ml::ops::MatrixMultiply<TypeParam> op;
  std::vector<TypeParam>                    backpropagatedSignals =
    op.Backward(std::vector<std::reference_wrapper<TypeParam const>>({a, b}), error);

  // test correct shapes
  ASSERT_EQ(backpropagatedSignals.size(), 2);
  ASSERT_EQ(backpropagatedSignals[0].shape(), (std::array<typename TypeParam::SizeType, 2>({1, 5})));
  ASSERT_EQ(backpropagatedSignals[1].shape(), (std::array<typename TypeParam::SizeType, 2>({5, 4})));

  // test correct values
  for (std::uint64_t i(0); i < inputGrad.size(); ++i)
    {
      EXPECT_EQ(backpropagatedSignals[0].Get(0, i), typename TypeParam::Type(inputGrad[i]));
    }
  for (std::uint64_t i(0); i < 5; ++i)
    {
      for (std::uint64_t j(0); j < 4; ++j)
	{
	  EXPECT_EQ(backpropagatedSignals[1].Get(i, j), typename TypeParam::Type(weightsGrad[i * 4 + j]));
	}
    }
}
