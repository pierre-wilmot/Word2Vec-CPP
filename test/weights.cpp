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

#include "weights.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

template <typename T>
class WeightsTest : public ::testing::Test
{
};

using MyTypes = ::testing::Types<int, float, double>;
TYPED_TEST_CASE(WeightsTest, MyTypes);

TYPED_TEST(WeightsTest, allocation_test)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  fetch::ml::ops::Weights<ArrayType, 2> w;
}

TYPED_TEST(WeightsTest, gradient_step_test)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;
  
  ArrayType        data({1, 8});
  ArrayType        error({1, 8});
  std::vector<int> dataInput({1, -2, 3, -4, 5, -6, 7, -8});
  std::vector<int> errorInput({-1, 2, 3, -5, -8, 13, -21, -34});
  std::vector<int> gtInput({2, -4, 0, 1, 13, -19, 28, 26});
  for (std::uint64_t i(0); i < 8; ++i)
  {
    data.Set(0, i, TypeParam(dataInput[i]));
    error.Set(0, i, TypeParam(errorInput[i]));
  }

  fetch::ml::ops::Weights<ArrayType, 2> w;
  w.SetData(data);
  ASSERT_EQ((w.fetch::ml::template Ops<ArrayType, 2>::Forward({})).Storage(), data.Storage());
  std::vector<ArrayType> errorSignal = w.Backward({}, error);
  w.Step(TypeParam(1));
  ArrayType output = w.fetch::ml::template Ops<ArrayType, 2>::Forward({});
  ASSERT_EQ(output.Storage(), data.Storage());  
  for (std::uint64_t i(0); i < 8; ++i)
    {
      EXPECT_FLOAT_EQ(output.Get(0, i), TypeParam(gtInput[i]));
    }    
}

TYPED_TEST(WeightsTest, stateDict)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;

  fetch::ml::ops::Weights<ArrayType, 2> w;
  fetch::ml::StateDict<ArrayType>       sd = w.StateDict();

  EXPECT_TRUE(sd.weights_ == nullptr);
  EXPECT_TRUE(sd.dict_.empty());

  ArrayType data({1, 8});
  w.SetData(data);
  sd = w.StateDict();
  EXPECT_EQ(sd.weights_->Storage(), data.Storage());
  EXPECT_TRUE(sd.dict_.empty());
}

TYPED_TEST(WeightsTest, loadStateDict)
{
  using ArrayType = fetch::math::Tensor<TypeParam, 2>;

  fetch::ml::ops::Weights<ArrayType, 2> w;

  std::shared_ptr<ArrayType>      data = std::make_shared<ArrayType>(std::array<typename ArrayType::SizeType, 2>({1, 8}));
  fetch::ml::StateDict<ArrayType> sd;
  sd.weights_ = data;
  w.LoadStateDict(sd);
  EXPECT_EQ((w.fetch::ml::template Ops<ArrayType, 2>::Forward({})).Storage(), data->Storage());
}
