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

#ifndef REVOLVEBRAIN_BRAIN_LEARNER_CPPNNEAT_TYPES_H_
#define REVOLVEBRAIN_BRAIN_LEARNER_CPPNNEAT_TYPES_H_

#include <boost/shared_ptr.hpp>

namespace CPPNEAT {
class Gene;

class Neuron;

class NeuronGene;

class ConnectionGene;

class GeneticEncoding;

class Mutator;

typedef boost::shared_ptr<Gene> GenePtr;

typedef boost::shared_ptr<Neuron> NeuronPtr;

typedef boost::shared_ptr<NeuronGene> NeuronGenePtr;

typedef boost::shared_ptr<ConnectionGene> ConnectionGenePtr;

typedef boost::shared_ptr<GeneticEncoding> GeneticEncodingPtr;

typedef boost::shared_ptr<Mutator> MutatorPtr;
}

#endif // REVOLVEBRAIN_BRAIN_LEARNER_CPPNNEAT_TYPES_H_
