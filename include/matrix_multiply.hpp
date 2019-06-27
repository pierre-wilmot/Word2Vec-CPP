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
#include "matrix_operations.hpp"

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class MatrixMultiply : public fetch::ml::BatchOps<T, 2>
{
public:
  using ArrayType      = T;
  using SizeType       = typename ArrayType::SizeType;

  MatrixMultiply()  = default;
  ~MatrixMultiply() = default;

  ArrayType Forward(std::vector<std::reference_wrapper<ArrayType const>> const &inputs,
                    ArrayType &                                                 output)
  {
    (void)output;
    assert(inputs.size() == 2);
    assert(inputs.at(0).get().shape().size() == 2);
    assert(inputs.at(1).get().shape().size() == 2);
    assert(output.shape() == ComputeOutputShape(inputs));

    fetch::math::Dot(inputs[0].get(), inputs[1].get(), output);
    return output;
  }

  std::vector<ArrayType> Backward(
      std::vector<std::reference_wrapper<const ArrayType>> const &inputs,
      ArrayType const &                                           errorSignal,
      std::vector<ArrayType>                                     &output)
  {
    assert(inputs.size() == 2 && output.size() == 2);

    output[0].Fill(0);
    output[1].Fill(0);
    
    fetch::math::DotTranspose(errorSignal, inputs.at(1).get(), output[0]);
    fetch::math::TransposeDot(inputs.at(0).get(), errorSignal, output[1]);

    return output;
  }

  std::array<SizeType, 2> ComputeOutputShape(
      std::vector<std::reference_wrapper<ArrayType const>> const &inputs) const
  {
    return {inputs.at(0).get().shape()[0], inputs.at(1).get().shape()[1]};
  }

  static constexpr char const *DESCRIPTOR = "MatrixMultiply";
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
