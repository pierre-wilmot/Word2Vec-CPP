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


namespace fetch {
  namespace math {

    template <typename T, typename SizeType, std::uint64_t RANK>
    class TensorIterator
    {
      friend class Tensor<T, RANK>;

    private:
      TensorIterator(std::array<SizeType, RANK> const &shape, std::array<SizeType, RANK> const &strides,
		     std::array<SizeType, RANK> const &padding, SizeType const &coordinate,
		     std::shared_ptr<T> const &storage, SizeType const &offset)
	: shape_(shape)
	, strides_(strides)
	, padding_(padding)
      {
	pointer_          = storage.get() + offset;
	original_pointer_ = storage.get() + offset;
	coordinate_.fill(0);
	coordinate_[0] = coordinate;
      }
		      
    public:
      bool operator!=(const TensorIterator<T, SizeType, RANK> &other) const
      {
	return !(*this == other);
      }

      bool operator==(const TensorIterator<T, SizeType, RANK> &other) const
      {
	return (original_pointer_ == other.original_pointer_) && (coordinate_ == other.coordinate_);
      }

      template<std::uint64_t N = RANK> // Specialise for RANK == 1, since we can skip the increment logic
      typename std::enable_if<(N == 1), TensorIterator&>::type operator++()
      {
	pointer_ += strides_[RANK-1];
	coordinate_[RANK-1]++;
	return *this;
      }
      
      template<std::uint64_t N = RANK>
      typename std::enable_if<(N > 1), TensorIterator&>::type operator++()
      {
	pointer_ += strides_[RANK-1];
	coordinate_[RANK-1]++;
	
	if (coordinate_[RANK-1] < shape_[RANK-1])
	  {
	    return *this;
	  }

	
	std::uint64_t i{RANK-1};
	while (i > 0 && (coordinate_[i] >= shape_[i]))
	  {
	    coordinate_[i] = 0;
	    coordinate_[i - 1] += 1;
	    i--;
	  }
	pointer_ = original_pointer_;
	for (std::size_t i(0); i < RANK-1 ; ++i)
	  {
	    pointer_ += coordinate_[i] * strides_[i];
	  }
	return *this;
      }

      T &operator*() const
      {
	return *pointer_;
      }

    private:
      const std::array<SizeType, RANK> &shape_;
      const std::array<SizeType, RANK> &strides_;
      const std::array<SizeType, RANK> &padding_;
      std::array<SizeType, RANK>       coordinate_;
      T *                              pointer_;
      T *                              original_pointer_;
    };

  }  // namespace math
}  // namespace fetch
