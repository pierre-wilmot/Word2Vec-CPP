#pragma once
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

#include "ops.hpp"

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class Sigmoid : public fetch::ml::ElementWiseOps<T, 2>
{
public:
  using ArrayType    = T;
  using DataType     = typename ArrayType::Type;
  using SizeType     = typename ArrayType::SizeType;
  using ArrayPtrType = std::shared_ptr<ArrayType>;

  Sigmoid()          = default;
  virtual ~Sigmoid() = default;

  virtual ArrayType Forward(std::vector<std::reference_wrapper<ArrayType const>> const &inputs,
                            ArrayType &                                                 output)
  {
    assert(inputs.size() == 1);
    assert(output.shape() == this->ComputeOutputShape(inputs));

    auto input_it = inputs.at(0).get().begin();
    auto input_end = inputs.at(0).get().end();
    auto output_it = output.begin();

    while (input_it != input_end)
      {
	*output_it = 1 / (1 + std::exp(*input_it * -1));
	++input_it;
	++output_it;
      }    
    return output;
  }

  virtual std::vector<ArrayType> Backward(
      std::vector<std::reference_wrapper<const ArrayType>> const &inputs,
      ArrayType const &                                           errorSignal)
  {
    assert(inputs.size() == 1);
    assert(inputs.front().get().shape() == errorSignal.shape());
    ArrayType returnSignal{errorSignal.shape()};
    ArrayType t{inputs.front().get().shape()};

    // gradient of sigmoid function is s(x)(1 - s(x))
    this->Forward(inputs, t);
    returnSignal.Fill(1);
    returnSignal.InlineSubtract(t);
    returnSignal.InlineMultiply(t);
    returnSignal.InlineMultiply(errorSignal);
    return {returnSignal};
  }

  static constexpr char const *DESCRIPTOR = "Sigmoid";
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
