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

#include "sigmoid.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

template <typename T>
class SigmoidTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<float>;

TYPED_TEST_CASE(SigmoidTest, MyTypes);

TYPED_TEST(SigmoidTest, forward_test)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  
  ArrayType           data({1, 8});
  std::vector<double> dataInput({1, -2, 3, -4, 5, -6, 7, -8});
  std::vector<double> gtInput({0.7310586, 0.1192029, 0.952574, 0.01798620996, 0.993307149,
                               0.002472623156635, 0.999088948806, 0.000335350130466});
  for (std::uint64_t i(0); i < 8; ++i)
  {
    data.Set(0, i, TypeParam(dataInput[i]));
  }
  fetch::ml::ops::Sigmoid<ArrayType> op;
  ArrayType                          prediction = op.fetch::ml::template Ops<ArrayType, 2>::Forward({std::cref(data)});

  // test correct values
  for (std::uint64_t i(0); i < 8; ++i)
  {
    EXPECT_FLOAT_EQ(prediction.Get(0, i), TypeParam(gtInput[i]));
  }
}

TYPED_TEST(SigmoidTest, backward_test)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;

  ArrayType           data({1, 8});
  ArrayType           error({1, 8});
  std::vector<double> dataInput({1, -2, 3, -4, 5, -6, 7, -8});
  std::vector<double> errorInput({0, 0, 0, 0, 1, 0, 0, 0});
  std::vector<double> gtInput({0, 0, 0, 0, 0.0066480329, 0, 0, 0});
  for (std::uint64_t i(0); i < 8; ++i)
  {
    data.Set(0, i, TypeParam(dataInput[i]));
    error.Set(0, i, TypeParam(errorInput[i]));
  }
  fetch::ml::ops::Sigmoid<ArrayType> op;
  std::vector<ArrayType>             prediction = op.fetch::ml::template Ops<ArrayType, 2>::Backward({data}, error);

  // test correct values
  for (std::uint64_t i(0); i < 8; ++i)
  {
    EXPECT_FLOAT_EQ(prediction[0].Get(0, i), gtInput[i]);
  }
}
