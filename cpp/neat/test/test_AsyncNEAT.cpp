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

#include <limits>
#include <string>
#include <vector>

#include "neat/AsyncNEAT.h"
#include "network/cpu/cpunetwork.h"

#include "test_AsyncNEAT.h"

const std::string test_name = "TestAsyncNeat";

TestAsyncNeat::TestAsyncNeat()
{
}

TestAsyncNeat::~TestAsyncNeat()
{
}

bool TestAsyncNeat::test()
{
  if (not testXOR())
  {
    return false;
  }

  return true;
}

// 0 0 -> 0
// 0 1 -> 1
// 1 0 -> 1
// 1 1 -> 0
bool TestAsyncNeat::testXOR()
{
  AsyncNeat::Init(test_name);
  AsyncNeat::SetSearchType(NEAT::GeneticSearchType::BLENDED);
  AsyncNeat::SetPopulationSize(10);
  AsyncNeat neat(2, 1, 1, test_name);
  float success_margin_error = 0.0001;

  bool success = false;
  float min_error = std::numeric_limits<float>().max();
  std::vector<float> inputs0 = {0, 0, 1, 1};
  std::vector<float> inputs1 = {0, 1, 0, 1};
  std::vector<float> expectedOutputs = {0, 1, 1, 0};

  int gen;
  for (gen = 1; gen < MAX_EVALUATIONS; gen++)
  {
    std::shared_ptr<NeatEvaluation> eval = neat.Evaluation();
    const NEAT::Organism *organism = eval->Organism();
    NEAT::CpuNetwork *net = reinterpret_cast< NEAT::CpuNetwork *> (
            organism->net.get());

    float error = 0;
    for (size_t test = 0; test < inputs0.size(); test++)
    {
      net->load_sensor(0, inputs0[test]);
      net->load_sensor(1, inputs1[test]);

      net->activate(1);
      NEAT::real_t *outputs = net->Outputs();
      error += std::abs(outputs[0] - expectedOutputs[test]);
    }

    if (min_error > error)
    {
      min_error = error;
    }

    eval->finish(1 / error);

    if (min_error < success_margin_error)
    {
      std::cout << "\nAfter "<< gen
                << " tries, a successful organism was found with an error of "
                << min_error << std::endl;
      std::cout << "The organism fitness is "
                << neat.Fittest()->Organism()->eval.fitness << std::endl;
      success = true;
      break;
    }
  }

  neat.CleanUp();
  return success;
}

int main()
{
  TestAsyncNeat t;
  return t.test() ? 0 : 1;
}
