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

#include "NEATEvaluation.h"

NeatEvaluation::NeatEvaluation(NEAT::Organism *organism)
        : organism(organism)
        , finished_callback(nullptr)
        , fitness(-1)
{}

void NeatEvaluation::finish(float fitness)
{
  organism->eval.fitness = fitness;

  if (not finished_callback)
  {
    std::cerr
            << "NeatEvaluation::finish() error, finish callback not setted!"
            << std::endl;
    return;
  }

  finished_callback(fitness);
}
