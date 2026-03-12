/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_views.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_sendnn_utils.h"
#include "spyre_storage_impl.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace spyre {

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Vec& sizes, const Vec& strides,
    SpyreTensorLayout device_layout) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset());
  self_tmp_->set_sizes_and_strides(sizes, strides);
  static_cast<SpyreTensorImpl*>(self_tmp_)->spyre_layout = device_layout;
  return self_;
}

// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and
// SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides, SpyreTensorLayout device_layout) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  self_.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides,
                                                     self.sym_storage_offset());
  static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl())->spyre_layout =
      device_layout;
  return self_;
}

// A group maps a set of old host dims to a set of new host dims.
// The product of sizes on each side must be equal.
struct DimGroup {
  std::vector<size_t> old_dims;
  std::vector<size_t> new_dims;
};

static inline at::Tensor spyre_view_impl(const at::Tensor& self,
                                         c10::IntArrayRef size) {
  c10::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  SpyreTensorLayout stl =
      static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl())->spyre_layout;
  return spyre_alias_with_sizes_and_strides(self, inferred_size, *stride, stl);
}

at::Tensor spyre_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre__unsafe_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

// Similar to as_strided with the following differences
// - offset is added to the existing offset (rather than replacing it)
// - view tracking is disabled similar to unsafe_view
at::Tensor spyre_reinterpret_tensor(const at::Tensor& self,
                                    c10::IntArrayRef size,
                                    c10::IntArrayRef stride,
                                    int64_t offset_increment) {
  SpyreTensorLayout stl =
      static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl())->spyre_layout;
  at::Tensor self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  self_tmp_->set_sizes_and_strides(size, stride);
  self_tmp_->spyre_layout = stl;
  return self_;
}

at::Tensor spyre_alias(const at::Tensor& self) {
  SpyreTensorLayout old_stl =
      static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl())->spyre_layout;
  return spyre_alias_with_sizes_and_strides(self, self.sym_sizes(),
                                            self.sym_strides(), old_stl);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view", TORCH_FN(spyre_view));
  m.impl("_unsafe_view", TORCH_FN(spyre__unsafe_view));
  m.impl("reinterpret_tensor", TORCH_FN(spyre_reinterpret_tensor));
  m.impl("alias", TORCH_FN(spyre_alias));
}

}  // namespace spyre
