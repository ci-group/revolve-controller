/*
* Copyright (C) 2017 Vrije Universiteit Amsterdam
*
* Licensed under the Apache License, Version 2.0 (the "License");
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Description: TODO: <Add brief description about file purpose>
* Author: TODO <Add proper author>
*
*/

#include <assert.h>
#include <map>
#include <string>
#include <vector>

#include "staticexperiment.h"

using namespace NEAT;

static std::vector< Test >
create_parallel_output_tests(
        std::string syms,
        std::vector< std::string > &sequences);

static struct SequenceInit
{
  SequenceInit()
  {
    create_static_experiment("seq-1bit-2el",
                             []()
                             {
                               std::string syms = "ab";
                               std::vector< std::string >
                                       seqs = permute_repeat(syms, 2);
                               return create_parallel_output_tests(syms, seqs);
                             });

    create_static_experiment("seq-1bit-3el",
                             []()
                             {
                               std::string syms = "ab";
                               std::vector< std::string >
                                       seqs = permute_repeat(syms, 3);
                               return create_parallel_output_tests(syms, seqs);
                             });

    create_static_experiment("seq-1bit-4el",
                             []()
                             {
                               std::string syms = "ab";
                               std::vector< std::string >
                                       seqs = permute_repeat(syms, 4);
                               return create_parallel_output_tests(syms, seqs);
                             });

    create_static_experiment("seq-1bit-5el",
                             []()
                             {
                               std::string syms = "ab";
                               std::vector< std::string >
                                       seqs = permute_repeat(syms, 5);
                               return create_parallel_output_tests(syms, seqs);
                             });
  }
} init;

static std::vector< Test > create_parallel_output_tests(
        std::string syms,
        std::vector< std::string > &sequences)
{
  const real_t weight_seq = 5;
  const real_t weight_query = 50;

  assert(syms.size() > 1);
  assert(sequences.size() > 1);
  for (size_t i = 1; i < sequences.size(); i++)
  {
    assert(sequences[0].size() == sequences[i].size());
  }

  size_t sequence_len = sequences[0].size();
  size_t nsyms = syms.size();
  size_t nbits = ceil(log2(nsyms));

  std::map< char, std::vector< real_t>> sym_encoding;
  // Create binary encoding for each symbol
  for (size_t i = 0; i < syms.size(); i++)
  {
    char sym = syms[i];
    assert(sym_encoding.find(sym) == sym_encoding.end());
    std::vector< real_t > &encoding = sym_encoding[sym];
    for (size_t bit = nbits; bit > 0; bit--)
    {
      if (i & (1 << (bit - 1)))
      {
        encoding.push_back(1.0);
      }
      else
      {
        encoding.push_back(0.0);
      }
    }
  }

  const real_t _ = 0.0;
  const real_t X = 1.0;

  std::vector< Test > tests;
  for (std::string &sequence: sequences)
  {
    std::vector< Step > steps;

    // Present sequence
    for (char sym: sequence)
    {
      // Create step in which symbol is presented
      {
        std::vector< real_t > input;
        // Symbol being provided in this step
        append(input, X);
        // Not querying
        append(input, _);
        append(input, sym_encoding[sym]);

        std::vector< real_t > output;
        // Empty output
        append(output, _, sequence_len * nbits);

        steps.emplace_back(input, output, weight_seq);
      }

      // Create silence
      {
        std::vector< real_t > input;
        // No symbol this step
        append(input, _);
        // Not querying
        append(input, _);
        // Empty symbol
        append(input, _, nbits);

        std::vector< real_t > output;
        // Empty output
        append(output, _, sequence_len * nbits);

        steps.emplace_back(input, output, weight_seq);
      }
    }

    // Query
    {
      std::vector< real_t > input;
      // No symbol
      append(input, _);
      // Querying
      append(input, X);
      // Empty symbol
      append(input, _, nbits);

      std::vector< real_t > output;
      for (char sym: sequence)
      {
        append(output, sym_encoding[sym]);
      }

      steps.emplace_back(input, output, weight_query);
    }

    tests.emplace_back(sequence, steps);
  }

  return tests;
}
