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

"""Tests for sentence_breaking_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.platform import test
from tensorflow_text.python.ops import sentence_breaking_ops_v2


# @test_util.run_all_in_graph_and_eager_modes
class SentenceFragmenterTestCasesV2(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          test_description="Test acronyms",
          #                1         2         3
          #      012345678901234567890123456789012345678
          doc=[["Welcome to the U.S. don't be surprised."]],
          expected_fragment_text=[[
              b"Welcome to the U.S.", b"don't be surprised."
          ]],
          expected_fragment_start=[[0, 20]],
          expected_fragment_end=[[19, 39]],
      ),
      dict(
          test_description="Test batch containing acronyms",
          #                1         2         3
          #      012345678901234567890123456789012345678      012345678
          doc=[["Welcome to the U.S. don't be surprised."], ["I.B.M. yo"]],
          expected_fragment_text=[[
              b"Welcome to the U.S.", b"don't be surprised."
          ], [b"I.B.M.", b"yo"]],
          expected_fragment_start=[[0, 20], [0, 7]],
          expected_fragment_end=[[19, 39], [6, 9]],
      ),
      dict(
          test_description="Test semicolons",
          #                1         2         3
          #      01234567890123456789012345678901234567
          doc=[["Welcome to the US; don't be surprised."]],
          expected_fragment_text=[[b"Welcome to the US; don't be surprised."]],
          expected_fragment_start=[[0]],
          expected_fragment_end=[[38]],
      ),
      dict(
          test_description="Basic test",
          #                1
          #      012345678901234
          doc=[["Hello. Foo bar!"]],
          expected_fragment_text=[[b"Hello.", b"Foo bar!"]],
          expected_fragment_start=[[0, 7]],
          expected_fragment_end=[[6, 15]],
      ),
      dict(
          test_description="Basic ellipsis test",
          #                1
          #      012345678901234
          doc=[["Hello...foo bar"]],
          expected_fragment_text=[[b"Hello...", b"foo bar"]],
          expected_fragment_start=[[0, 8]],
          expected_fragment_end=[[8, 15]],
      ),
      dict(
          test_description="Parentheses and ellipsis test",
          #                1         2
          #      012345678901234567890123456789
          doc=[["Hello (who are you...) foo bar"]],
          expected_fragment_text=[[b"Hello (who are you...)", b"foo bar"]],
          expected_fragment_start=[[0, 23]],
          expected_fragment_end=[[22, 30]],
      ),
      dict(
          test_description="Punctuation after parentheses test",
          #                1         2
          #      01234567890123456789012345678
          doc=[["Hello (who are you)? Foo bar!"]],
          expected_fragment_text=[[b"Hello (who are you)?", b"Foo bar!"]],
          expected_fragment_start=[[0, 21]],
          expected_fragment_end=[[20, 29]],
      ),
      dict(
          test_description="MidFragment Parentheses test",
          #                1         2         3
          #      0123456789012345678901234567890123
          doc=[["Hello (who are you) world? Foo bar"]],
          expected_fragment_text=[[b"Hello (who are you) world?", b"Foo bar"]],
          expected_fragment_start=[[0, 27]],
          expected_fragment_end=[[26, 34]],
      ),
      dict(
          test_description="Many final punctuation test",
          #                1         2
          #      012345678901234567890123
          doc=[["Hello!!!!! Who are you??"]],
          expected_fragment_text=[[b"Hello!!!!!", b"Who are you??"]],
          expected_fragment_start=[[0, 11]],
          expected_fragment_end=[[10, 24]],
      ),
      dict(
          test_description="Test emoticons within text",
          #                1         2
          #      0123456789012345678901234
          doc=[["Hello world :) Oh, hi :-O"]],
          expected_fragment_text=[[b"Hello world :)", b"Oh, hi :-O"]],
          expected_fragment_start=[[0, 15]],
          expected_fragment_end=[[14, 25]],
      ),
      dict(
          test_description="Test emoticons with punctuation following",
          #                1
          #      0123456789012345678
          doc=[["Hello world :)! Hi."]],
          expected_fragment_text=[[b"Hello world :)!", b"Hi."]],
          expected_fragment_start=[[0, 16]],
          expected_fragment_end=[[15, 19]],
      ),
      dict(
          test_description="Test emoticon list",
          #                1
          #      01234 56789012345678
          doc=[[":) :-\\ (=^..^=) |-O"]],
          expected_fragment_text=[[b":)", b":-\\", b"(=^..^=)", b"|-O"]],
          expected_fragment_start=[[0, 3, 7, 16]],
          expected_fragment_end=[[2, 6, 15, 19]],
      ),
      dict(
          test_description="Test emoticon batch",
          #      01      01 2      01234567      012
          doc=[[":)"], [":-\\"], ["(=^..^=)"], ["|-O"]],
          expected_fragment_text=[[b":)"], [b":-\\"], [b"(=^..^=)"], [b"|-O"]],
          expected_fragment_start=[[0], [0], [0], [0]],
          expected_fragment_end=[[2], [3], [8], [3]],
      ),
  ])
  def testHeuristicBasedSentenceBreaker(self, test_description, doc,
                                        expected_fragment_text,
                                        expected_fragment_start,
                                        expected_fragment_end):
    sentence_breaker = sentence_breaking_ops_v2.HeuristicBasedSentenceBreaker()

    fragments = self.evaluate(
        sentence_breaker.break_sentences_with_offsets(doc))
    fragment_text, fragment_starts, fragment_ends = fragments

    self.assertAllEqual(expected_fragment_text, fragment_text)
    self.assertAllEqual(expected_fragment_start, fragment_starts)
    self.assertAllEqual(expected_fragment_end, fragment_ends)


if __name__ == "__main__":
  test.main()
