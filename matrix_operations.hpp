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
  
  template <typename ArrayType>
  void Dot(ArrayType const &A, ArrayType const &B, ArrayType &ret)
  {
    for (typename ArrayType::SizeType i(0); i < A.shape()[0]; ++i)
      {
	for (typename ArrayType::SizeType j(0); j < B.shape()[1]; ++j)
	  {
	    ret.Set(i, j, A.Get(i, 0) * B.Get(0, j));
	    for (typename ArrayType::SizeType k(1); k < A.shape()[1]; ++k)
	      {
		ret.Set(i, j, ret.Get(i, j) + A.Get(i, k) * B.Get(k, j));
	      }
	  }
      }
  }
  
  template <class ArrayType>
  void DotTranspose(ArrayType const &A, ArrayType const &B, ArrayType &ret)
  {
    for (size_t i(0); i < A.shape()[0]; ++i)
      {
	for (size_t j(0); j < B.shape()[0]; ++j)
	  {
	    for (size_t k(0); k < A.shape()[1]; ++k)
	      {
		ret.Set(i, j, ret.Get(i, j) + A.Get(i, k) * B.Get(j, k));
	      }
	  }
      }
  }
  
  template <class ArrayType>
  void TransposeDot(ArrayType const &A, ArrayType const &B, ArrayType &ret)
  {
    for (size_t i(0); i < A.shape()[1]; ++i)
      {
	for (size_t j(0); j < B.shape()[1]; ++j)
	  {
	    for (size_t k(0); k < A.shape()[0]; ++k)
	      {
		ret.Set(i, j, ret.Get(i, j) + A.Get(k, i) * B.Get(k, j));
	      }
	  }
      }
  }

}  // namespace math
}  // namespace fetch
