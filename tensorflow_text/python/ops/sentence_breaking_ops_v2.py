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

"""Break sentence ops."""
import abc

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_sentence_breaking_ops_v2 = load_library.load_op_library(resource_loader.get_path_to_datafile('_sentence_breaking_ops_v2.so'))
from tensorflow_text.python.ops import regex_split_ops


class SentenceBreaker(object):
  """An abstract base class for sentence breaker implementations."""

  @abc.abstractmethod
  def break_sentences(self, input):  # pylint: disable=redefined-builtin
    """Splits `input` into sentences.

    Args:
       input: A string `Tensor` of shape [batch] with a batch of documents.

    Returns:
       A string `RaggedTensor` of shape [batch, (num_sentences)] with each input
       broken up into its constituent sentences.
    """
    raise NotImplementedError()


class SentenceBreakerWithOffsets(SentenceBreaker):
  """An abstract base class for sentence breakers that support offsets."""

  @abc.abstractmethod
  def break_sentences_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Splits `input` into sentences and returns the starting & ending offsets.

    Args:
      input: A string `Tensor` of shape [batch] with a batch of documents.

    Returns:
      A tuple of (sentences, begin_offset, end_offset) where:

      sentences: A string `RaggedTensor` of shape [batch, (num_sentences)] with
        each input broken up into its constituent sentences.
      begin_offset: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the inclusive beginning byte offset of a sentence.
      end_offset: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the exclusive ending byte offset of a sentence.
    """
    raise NotImplementedError()


class RegexSentenceBreaker(SentenceBreakerWithOffsets):
  """A `SentenceBreaker` that splits sentences separated by a newline.

  `RegexSentenceBreaker` splits text when a newline character is detected.
  The newline character is determined by a regex pattern. It also returns the
  sentence beginning and ending byte offsets as well.
  """

  def __init__(self, new_sentence_regex=None):
    r"""Creates an instance of `RegexSentenceBreaker`.

    Args:
      new_sentence_regex: (optional) A string containing the regex pattern of a
        new line sentence delimiter. Default is '\r?\n'.
    """
    if not new_sentence_regex:
      new_sentence_regex = '\r?\n'
    self._new_sentence_regex = new_sentence_regex

  def break_sentences(self, input):  # pylint: disable=redefined-builtin
    return regex_split_ops.regex_split(input, self._new_sentence_regex)

  def break_sentences_with_offsets(self, input):  # pylint: disable=redefined-builtin
    return regex_split_ops.regex_split_with_offsets(input,
                                                    self._new_sentence_regex)


class HeuristicBasedSentenceBreaker(SentenceBreakerWithOffsets):
  """A `SentenceBreaker` that splits sentence fragments separated by punctuation.

  It returns the text of each sentence fragment, the starting index of the
  fragment, and the ending index (exclusive) of the fragment.
  """

  def break_sentences(self, doc):  # pylint: disable=redefined-builtin
    results, _, _ = self.break_sentences_with_offsets(doc)
    return results

  def break_sentences_with_offsets(self, doc):  # pylint: disable=redefined-builtin
    """Splits `doc` into sentence fragments, returns text, start & end offsets.

    Args:
      doc: A string `Tensor` of shape [batch] with a batch of documents.

    Returns:
      A tuple of (fragment_text, start, end) where:

      fragment_text: A string `RaggedTensor` of shape [batch, (num_sentences)]
      with each input broken up into its constituent sentence fragments.
      start: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the inclusive beginning byte offset of a sentence.
      end: A int64 `RaggedTensor` of shape [batch, (num_sentences)]
        where each entry is the exclusive ending byte offset of a sentence.
    """
    if not isinstance(doc, ragged_tensor.RaggedTensor):
      doc = ragged_tensor.RaggedTensor.from_tensor(doc)

    # Run sentence fragmenter op v2
    fragment = gen_sentence_breaking_ops_v2.sentence_fragments_v2(
        doc=doc.flat_values)
    start, end, properties, terminal_punc_token, row_lengths = fragment

    # Pack and create `RaggedTensor`s
    start, end, properties, terminal_punc_token = tuple(
        ragged_tensor.RaggedTensor.from_row_lengths(value, row_lengths)
        for value in [start, end, properties, terminal_punc_token])

    # Extract fragment text using offsets
    def _substring(x):
      s, pos, length = x
      return string_ops.substr(s, pos, length)

    fragment_text = ragged_map_ops.map_fn(
        _substring, (doc, start, math_ops.subtract(end, start)),
        infer_shape=False,
        dtype=dtypes.string)

    return fragment_text, start, end
