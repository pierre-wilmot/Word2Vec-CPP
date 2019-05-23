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

#include "weights.hpp"
#include <set>

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class Embeddings : public fetch::ml::ops::Weights<T, 2>
{
public:
  using ArrayType    = T;
  using DataType     = typename ArrayType::Type;
  using ArrayPtrType = std::shared_ptr<ArrayType>;
  using SizeType     = typename ArrayType::SizeType;

  Embeddings(SizeType dataPoints, SizeType dimensions)
  {
    ArrayType weights = ArrayType({dataPoints, dimensions});
    fetch::ml::ops::Weights<ArrayType, 2>::Initialise(weights, dataPoints, dimensions);
    this->SetData(weights);
  }

  Embeddings(ArrayType &weights)
  {
    this->SetData(weights);
  }

  virtual ~Embeddings() = default;

  virtual ArrayType Forward(std::vector<std::reference_wrapper<ArrayType const>> const &inputs,
                            ArrayType &                                                 output)
  {
    assert(this->output_);
    assert(inputs.size() == 1);
    assert(output.shape() == ComputeOutputShape(inputs));

    uint64_t j(0);
    for (DataType const &i : inputs.front().get())
    {
      output.Slice(j).Copy(this->output_->Slice(SizeType(i)));
      j++;
    }
    return output;
  }

  virtual std::vector<ArrayType> Backward(
      std::vector<std::reference_wrapper<ArrayType const>> const &inputs,
      ArrayType const &                                           errorSignal)
  {
    assert(inputs.size() == 1);

    uint64_t j(0);
    for (DataType const &i : inputs.front().get())
    {
      updated_rows_.insert(typename ArrayType::SizeType(double(i)));
      this->gradient_accumulation_->Slice(typename ArrayType::SizeType(double(i))).InlineAdd(errorSignal.Slice(j));
      j++;
    }
    return {errorSignal};
  }

  virtual void Step(typename T::Type learningRate)
  {
    for (auto const &r : updated_rows_)
    {
      auto gradientAccumulationSlice = this->gradient_accumulation_->Slice(r);
      auto outputSlice               = this->output_->Slice(r);
      auto it1                       = gradientAccumulationSlice.begin();
      auto end                       = gradientAccumulationSlice.end();
      auto it2                       = outputSlice.begin();
      while (it1 != end)
      {
        *it2 += (*it1 * learningRate);
        *it1 = 0;
        ++it1;
        ++it2;
      }
    }
    updated_rows_.clear();
  }

  virtual std::array<SizeType, 2> ComputeOutputShape(
      std::vector<std::reference_wrapper<ArrayType const>> const &inputs) const
  {
    return {inputs.front().get().Size(), this->output_->shape()[1]};
  }

private:
  std::set<typename ArrayType::SizeType> updated_rows_;
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
