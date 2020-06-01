# coding=utf-8
# Copyright 2020 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""Tensorflow lowercasing operation for UTF8 strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_normalize_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_normalize_ops.so'))


# pylint: disable=redefined-builtin
def case_fold_utf8(input, name=None):
  """Applies case folding to every UTF-8 string in the input.

  The input is a `Tensor` or `RaggedTensor` of any shape, and the resulting
  output has the same shape as the input. Note that NFKC normalization is
  implicitly applied to the strings.

  For example:

  ```python
  >>> case_fold_utf8(['The   Quick-Brown',
  ...                 'CAT jumped over',
  ...                 'the lazy dog  !!  ']
  tf.Tensor(['the   quick-brown' 'cat jumped over' 'the lazy dog  !!  '],
            shape=(3,), dtype=string)
  ```

  Args:
    input: A `Tensor` or `RaggedTensor` of UTF-8 encoded strings.
    name: The name for this op (optional).

  Returns:
    A `Tensor` or `RaggedTensor` of type string, with case-folded contents.
  """
  with ops.name_scope(name, "CaseFoldUTF8", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result = gen_normalize_ops.case_fold_utf8(input_tensor.flat_values)
      return input_tensor.with_flat_values(result)
    else:
      return gen_normalize_ops.case_fold_utf8(input_tensor)


# pylint: disable=redefined-builtin)
def normalize_utf8(input, normalization_form="NFKC", name=None):
  """Normalizes each UTF-8 string in the input tensor using the specified rule.

  See http://unicode.org/reports/tr15/

  Args:
    input: A `Tensor` or `RaggedTensor` of type string. (Must be UTF-8.)
    normalization_form: One of the following string values ('NFC', 'NFKC',
      'NFD', 'NFKD'). Default is 'NFKC'.
    name: The name for this op (optional).

  Returns:
    A `Tensor` or `RaggedTensor` of type string, with normalized contents.
  """
  with ops.name_scope(name, "NormalizeUTF8", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result = gen_normalize_ops.normalize_utf8(input_tensor.flat_values,
                                                normalization_form)
      return input_tensor.with_flat_values(result)
    else:
      return gen_normalize_ops.normalize_utf8(input_tensor, normalization_form)


# pylint: disable=redefined-builtin)
def normalize_utf8_with_offsets(input, normalization_form="NFKC", name=None):
  """Normalizes each UTF-8 string in the input tensor using the specified rule.

  Returns normalized strings and an offset map used by another operation to map
  post-normalized string offsets to pre-normalized string offsets.

  See http://unicode.org/reports/tr15/

  Args:
    input: A `Tensor` or `RaggedTensor` of type string. (Must be UTF-8.)
      normalization_form: One of the following string values ('NFC', 'NFKC').
        Default is 'NFKC'.
    name: The name for this op (optional).

  Returns:
    A tuple of (results, offset_map) where:

    results: `Tensor` or `RaggedTensor` of type string, with normalized
      contents.
    offset_map: `variant` used to map the post-normalized string offsets
      to pre-normalized string offsets.
  """
  with ops.name_scope(name, "NormalizeUTF8WithOffsets", [input]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input, dtype=dtypes.string)
    if ragged_tensor.is_ragged(input_tensor):
      result, offset_map = gen_normalize_ops.normalize_utf8_with_offsets(
          input_tensor.flat_values, normalization_form)
      return input_tensor.with_flat_values(result), offset_map
    else:
      return gen_normalize_ops.normalize_utf8_with_offsets(
          input_tensor, normalization_form)


# pylint: disable=redefined-builtin)
def find_source_offsets(offset_map, input_offsets, name=None):
  """Maps the input post-normalized string offsets to pre-normalized offsets.

  Returns the source (i.e. pre-normalized) string offsets mapped from the input
  post-normalized string offsets using the input offset_map, which is an output
  from the `normalize_utf8_with_offsets` op.

  For example:

  ```python
  >>> post_normalized_str, offset_map = normalize_utf8_with_offsets(
  ...                                 ["株式会社", "ＫＡＤＯＫＡＷＡ"])
  >>> find_source_offsets(offset_map, [[0, 1, 2], [0, 1]])
  tf.Tensor([[0, 1, 2], [0, 3]], shape=(2, 3), dtype=int64)
  ```

  Args:
    offset_map: `variant` used to map the post-normalized string offsets to
      pre-normalized string offsets.
    input_offsets: A `Tensor` or `RaggedTensor` of type int64 representing the
      the post-normalized string offsets,
    name: The name for this op (optional).

  Returns:
    results: `Tensor` or `RaggedTensor` of type int64, with pre-normalized
      string offsets content.
  """

  with ops.name_scope(name, "FindSourceOffsets", [input_offsets]):
    input_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input_offsets, dtype=dtypes.int64)

    if ragged_tensor.is_ragged(input_offsets_tensor):
      output_values = gen_normalize_ops.find_source_offsets(
          offset_map=offset_map,
          input_offsets_values=input_offsets_tensor.flat_values,
          input_offsets_splits=input_offsets_tensor.nested_row_splits[-1])
      output_offsets = input_offsets_tensor.with_flat_values(output_values)
      return output_offsets
    else:
      if input_offsets_tensor.shape.ndims > 1:
        output_offsets = find_source_offsets(
            offset_map, ragged_conversion_ops.from_tensor(input_offsets_tensor))
        return ragged_conversion_ops.to_tensor(output_offsets)
      elif input_offsets_tensor.shape.ndims == 0:
        output_offsets = find_source_offsets(
            offset_map, array_ops.expand_dims(input_offsets_tensor, 0))
        return output_offsets[0]
      else:
        output_offsets = find_source_offsets(
            offset_map, array_ops.expand_dims(input_offsets_tensor, 0))
        return array_ops.squeeze(output_offsets, [0])
