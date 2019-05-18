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

#include <cassert>
#include <cstdint>
#include <array>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string.h> // memset
#include <vector>

namespace fetch {
namespace math {
  template <typename T, std::uint64_t RANK>
  class Tensor;
}
}

#include "tensor_iterator.hpp"

namespace fetch {
namespace math {

  template <typename T, std::uint64_t RANK>
class Tensor
{
public:
  using Type                             = T;
  using SizeType                         = std::uint64_t;
  using SelfType                         = Tensor<T, RANK>;
  static const SizeType DefaultAlignment = 8;  // Arbitrary picked

  friend class Tensor<T, RANK+1>; // let's us access private member of slice (Tensor<T, RANK-1>)
  
public:
  Tensor(std::array<SizeType, RANK>           shape,
         std::array<SizeType, RANK>           strides = std::array<SizeType, RANK>{{SizeType(-1)}},
	 std::array<SizeType, RANK>           padding = std::array<SizeType, RANK>{{SizeType(-1)}},
         std::shared_ptr<T>              storage = nullptr, SizeType offset = 0)
    : storage_(std::move(storage))
    , offset_(offset)
  {
    assert(shape.size() == RANK);
    assert(padding.empty() || padding.size() == shape.size());
    assert(strides.empty() || strides.size() == shape.size());
          
    std::copy(shape.begin(), shape.end(), shape_.begin());
    if (strides[0] != SizeType(-1))
      {
	std::copy(strides.begin(), strides.end(), input_strides_.begin());
      }
    else
      {
	input_strides_.fill(1);
      }
    if (padding[0] != SizeType(-1))
      {
	std::copy(padding.begin(), padding.end(), padding_.begin());
      }
    else
      {
	padding_.fill(0);
	if (((input_strides_.back() * shape_.back()) % DefaultAlignment) != 0)
	  {
	    padding_[RANK - 1] = DefaultAlignment - ((input_strides_.back() * shape_.back()) % DefaultAlignment);
	  }
      }
    strides_[RANK-1] = input_strides_[RANK-1];
    if (RANK > 1)
      {
	for (std::uint64_t i(RANK-2) ; i > 0 ; --i)
	  {
	    strides_[i] = (strides_[i+1] * shape_[i+1] + padding_[i+1]) * input_strides_[i];
	  }
	strides_[0] = (strides_[1] * shape_[1] + padding_[1]) * input_strides_[0];
      }
      if (!storage_)
	{
	  offset_ = 0;
	  if (!shape_.empty())
	    {
	      storage_ = std::shared_ptr<T>(new T[Capacity()], std::default_delete<T[]>());
	      memset(static_cast<void*>(storage_.get()), 0, Capacity() * sizeof(T));
	    }
	}
      size_ = std::accumulate(shape_.begin(), shape_.end(), SizeType(1), std::multiplies<SizeType>());
  }

  Tensor(Tensor const &t)     = default;
  Tensor(Tensor &&t) noexcept = default;
  Tensor &operator=(Tensor const &other) = default;
  Tensor &operator=(Tensor &&) = default;

  /**
   * Returns a deep copy of this tensor
   * @return
   */
  SelfType Clone() const
  {
    SelfType copy(this->shape_);
    copy.Copy(*this);
    return copy;
  }

  /**
   * Copy data from another tensor into this one
   * assumes relevant stride/offset etc. are still applicable
   * @param other
   * @return
   */
  Tensor &Copy(SelfType const &o)
  {
    assert(Size() == o.Size());
    auto it1 = this->begin();
    auto end = this->end();
    auto it2 = o.begin();

    while (it1 != end)
    {
      *it1 = *it2;
      ++it1;
      ++it2;
    }
    return *this;
  }

  // TODO(private, 520) fix capitalisation (kepping it consistent with NDArray for now)
  std::array<SizeType, RANK> const &shape() const
  {
    return shape_;
  }

  std::array<SizeType, RANK> const &Strides() const
  {
    return input_strides_;
  }

  std::array<SizeType, RANK> const &Padding() const
  {
    return padding_;
  }

  SizeType Offset() const
  {
    return offset_;
  }

  SizeType DimensionSize(SizeType dim) const
  {
    if (!shape_.empty() && dim < shape_.size())
    {
      return strides_[dim];
    }
    return 0;
  }

  SizeType Capacity() const
  {
    return std::max(SizeType(1), DimensionSize(0) * shape_[0] + padding_[0]);
  }

  SizeType Size() const
  {
    return size_;
  }

  void Fill(T const &value)
  {
    for (T &e : *this)
    {
      e = value;
    }
  }

  /////////////////
  /// Iterators ///
  /////////////////

  TensorIterator<T, SizeType, RANK> begin() const  // Need to stay lowercase for range basedloops
  {
    return TensorIterator<T, SizeType, RANK>(shape_, strides_, padding_, 0, storage_, offset_);
  }

  TensorIterator<T, SizeType, RANK> end() const  // Need to stay lowercase for range basedloops
  {
    return TensorIterator<T, SizeType, RANK>(shape_, strides_, padding_, shape_[0], storage_, offset_);
  }

  //////////////////////////
  /// OFFSET COMPUTATION ///
  //////////////////////////

  template <SizeType N, typename FirstIndex, typename... Indices>
  constexpr SizeType OffsetForIndices(FirstIndex &&index, Indices &&... indices) const
  {
    //    static_assert(std::is_integral<typename std::remove_reference<FirstIndex>::type>::value, "Can't index tensor using non integer type");
    return static_cast<SizeType>(index) * strides_[N] + OffsetForIndices<N + 1>(std::forward<Indices>(indices)...);
  }
  
  template <SizeType N, typename FirstIndex>
  constexpr SizeType OffsetForIndices(FirstIndex &&index) const
  {
    //    static_assert(std::is_integral<typename std::remove_reference<FirstIndex>::type>::value, "Can't index tensor using non integer type");
    return static_cast<SizeType>(index) * strides_[N];
  }

  template <SizeType N, typename FirstIndex, typename... Indices>
  constexpr std::pair<SizeType, T> OffsetAndValueForIndices(FirstIndex &&index, Indices &&... indices) const
  {
    //    static_assert(std::is_integral<typename std::remove_reference<FirstIndex>::type>::value, "Can't index tensor using non integer type");
    return std::pair<SizeType, T>(OffsetAndValueForIndices<N + 1>(std::forward<Indices>(indices)...).first + static_cast<SizeType>(index) * strides_[N], OffsetAndValueForIndices<N + 1>(std::forward<Indices>(indices)...).second);
  }
  
  template <SizeType N, typename FirstIndex>
  constexpr std::pair<SizeType, T> OffsetAndValueForIndices(FirstIndex &&index) const
  {
    //    static_assert(std::is_integral<typename std::remove_reference<FirstIndex>::type>::value, "Can't index tensor using non integer type");
    return std::pair<SizeType, T>(0, index);
  }

  /////////////////
  /// ACCESSORS ///
  /////////////////
  
  template <typename... Indices>
  T const &Get(Indices... indices) const
  {
    static_assert(sizeof...(Indices) == RANK, "Number of indexes in Get() doesn't match tensor rank");
    assert(sizeof...(Indices) == shape_.size());
    return storage_.get()[offset_ + OffsetForIndices<0>(indices...)];
  }

  ///////////////
  /// SETTERS ///
  ///////////////

  template <typename... Indices>
  void Set(Indices... indicesAndValuesPack)
  {
    static_assert(sizeof...(Indices) == RANK + 1, "Number of indexes in Set() doesn't match tensor rank");
    assert(sizeof...(Indices) == shape_.size() + 1);
    std::pair<SizeType, T> ret = OffsetAndValueForIndices<0>(indicesAndValuesPack...);
    storage_.get()[offset_ + ret.first] = ret.second;
  }
  
  /*
   * return a slice of the tensor along the first dimension
   */
  Tensor<T, RANK-1> Slice(SizeType i) const
  {
    assert(shape_.size() > 1 && i < shape_[0]);

    std::array<SizeType, RANK-1> slice_shape;
    std::array<SizeType, RANK-1> slice_strides;
    std::array<SizeType, RANK-1> slice_padding;

    std::copy(std::next(shape_.begin()), shape_.end(), slice_shape.begin());
    std::copy(std::next(input_strides_.begin()), input_strides_.end(), slice_strides.begin());
    std::copy(std::next(padding_.begin()), padding_.end(), slice_padding.begin());
    
    Tensor<T, RANK-1> ret(slice_shape, slice_strides, slice_padding,
			  storage_, offset_ + i * DimensionSize(0));

    std::copy(std::next(strides_.begin()), strides_.end(), ret.strides_.begin());
    std::copy(std::next(padding_.begin()), padding_.end(), ret.padding_.begin());
    return ret;
  }

  std::shared_ptr<T> Storage() const
  {
    return storage_;
  }

  SelfType &InlineAdd(T const &o)
  {
    for (T &e : *this)
    {
      e += o;
    }
    return *this;
  }

  SelfType &InlineAdd(Tensor<T, RANK> const &o, T alpha = T(1.0f))
  {
    assert(Size() == o.Size());
    auto it1 = this->begin();
    auto end = this->end();
    auto it2 = o.begin();

    while (it1 != end)
    {
      
      *it1 += (*it2 * alpha);
      ++it1;
      ++it2;
    }
    return *this;
  }

  SelfType &InlineSubtract(T const &o)
  {
    for (T &e : *this)
    {
      e -= o;
    }
    return *this;
  }

  SelfType &InlineSubtract(Tensor<T, RANK> const &o)
  {
    assert(Size() == o.Size());
    auto it1 = this->begin();
    auto end = this->end();
    auto it2 = o.begin();

    while (it1 != end)
    {
      *it1 -= *it2;
      ++it1;
      ++it2;
    }
    return *this;
  }

  SelfType &InlineMultiply(T const &o)
  {
    for (T &e : *this)
    {
      e *= o;
    }
    return *this;
  }

  SelfType &InlineMultiply(Tensor<T, RANK> const &o)
  {
    assert(Size() == o.Size());
    auto it1 = this->begin();
    auto end = this->end();
    auto it2 = o.begin();

    while (it1 != end)
    {
      *it1 *= *it2;
      ++it1;
      ++it2;
    }
    return *this;
  }

  SelfType &InlineDivide(T const &o)
  {
    for (T &e : *this)
    {
      e /= o;
    }
    return *this;
  }

  SelfType &InlineDivide(Tensor<T, RANK> const &o)
  {
    assert(Size() == o.Size());
    auto it1 = this->begin();
    auto end = this->end();
    auto it2 = o.begin();

    while (it1 != end)
    {
      *it1 /= *it2;
      ++it1;
      ++it2;
    }
    return *this;
  }

  T Sum() const
  {
    return std::accumulate(begin(), end(), T(0));
  }

  SelfType Transpose() const
  {
    assert(shape_.size() == 2);
    SelfType ret({shape_[1], shape_[0]}, /* shape */
		 {SizeType(-1), SizeType(-1)},                       /* stride */
		 {SizeType(-1), SizeType(-1)},                       /* padding */
		 storage_, offset_);
    std::copy(strides_.rbegin(), strides_.rend(), ret.strides_.begin());
    std::copy(padding_.rbegin(), padding_.rend(), ret.padding_.begin());
    return ret;
  }



  template<std::uint64_t N = RANK>
  typename std::enable_if<(N == 1), std::string>::type ToString() const
  {
    std::stringstream ss;
    ss << std::setprecision(5) << std::fixed << std::showpos;
    for (SizeType i(0); i < shape_[0]; ++i)
      {
        ss << Get(i) << "\t";
      }
    return ss.str();
  }

  template<std::uint64_t N = RANK>
  typename std::enable_if<(N > 1), std::string>::type ToString() const
  {
    std::stringstream ss;
    for (SizeType i(0) ; i < shape_[0] ; ++i)
      {
	ss << Slice(i).ToString() << "\n";
      }
    return ss.str();
  }

private:
  std::array<SizeType, RANK>      shape_;
  std::array<SizeType, RANK>      padding_;
  std::array<SizeType, RANK>      strides_;
  std::array<SizeType, RANK>      input_strides_;
  std::shared_ptr<T>              storage_;
  SizeType                        offset_;
  SizeType                        size_;
};
}  // namespace math
}  // namespace fetch
