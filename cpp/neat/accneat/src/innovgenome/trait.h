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

/*
 Copyright 2001 The University of Texas at Austin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef _TRAIT_H_
#define _TRAIT_H_

#include <yaml-cpp/yaml.h>

#include "neat.h"
#include "util/rng.h"

namespace NEAT {
  const int NUM_TRAIT_PARAMS = 8;

  // ------------------------------------------------------------------
  // TRAIT: A Trait is a group of parameters that can be expressed
  //        as a group more than one time.  Traits save a genetic
  //        algorithm from having to search vast parameter landscapes
  //        on every node.  Instead, each node can simply point to a trait
  //        and those traits can evolve on their own
  class Trait {
    // ************ LEARNING PARAMETERS ***********
    // The following parameters are for use in
    //   neurons that learn through habituation,
    //   sensitization, or Hebbian-type processes

  public:
    /// \brief Used in file saving and loading
    int trait_id;

    /// \brief Keep traits in an array
    real_t params[NUM_TRAIT_PARAMS];

    /// \brief
    Trait();

    /// \brief
    Trait(int id, real_t p1, real_t p2, real_t p3, real_t p4, real_t p5,
          real_t p6, real_t p7, real_t p8, real_t p9);

    /// \brief Copy Constructor
    Trait(const Trait &t);

    /// \brief Create a trait exactly like another trait
    Trait(Trait *t);

    /// \brief Special constructor off a file assume word "trait" has been read
    /// in
    Trait(const char *argline);

    /// \brief Special Constructor creates a new Trait which is the average of 2
    /// existing traits passed in
    Trait(const Trait &t1, const Trait &t2);

    /// \brief Dump trait to a stream
    void print_to_file(std::ostream &outFile) const;

    /// \brief Perturb the trait parameters slightly
    void mutate(rng_t &rng);
  };
}  // namespace NEAT

namespace YAML {
  template<>
  struct convert<NEAT::Trait> {
    static Node encode(const NEAT::Trait &rhs);

    static bool decode(const Node &node, NEAT::Trait &rhs);
  };
}

#endif
