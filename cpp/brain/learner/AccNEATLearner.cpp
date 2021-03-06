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
* Author: Matteo De Carlo
 * Date: March 13, 2017
*
*/

#include <ctime>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "brain/controller/AccNEATCPPNController.h"

#include "AccNEATLearner.h"

using namespace revolve::brain;

AccNEATLearner::AccNEATLearner(
        const std::string &robot_name,
        EvaluatorPtr evaluator,
        size_t n_inputs,
        size_t n_outputs,
        const float evaluationTime,
        const long maxEvaluations
)
        : BaseLearner(std::unique_ptr< BaseController >(
        new AccNEATCPPNController(n_inputs, n_outputs)), robot_name)
        , evaluator(evaluator)
        , n_inputs(n_inputs)
        , n_outputs(n_outputs)
        , generation_counter(0)
        , start_eval_time(std::numeric_limits< double >::lowest())
        , MAX_EVALUATIONS(maxEvaluations)
        , EVALUATION_TIME(evaluationTime)
{
  this->initAsyncNeat();
}

AccNEATLearner::~AccNEATLearner()
{
  AsyncNeat::CleanUp();
}

void AccNEATLearner::initAsyncNeat()
{
  AsyncNeat::Init(robot_name);
  std::unique_ptr< AsyncNeat > neat(new AsyncNeat(
          (size_t)n_inputs,
          (size_t)n_outputs,
          (int)std::time(0),  // random seed,
          robot_name));
  this->neat = std::move(neat);
}

BaseController *AccNEATLearner::update(
        const std::vector< SensorPtr > &sensors,
        double t,
        double step)
{
  // Evaluate policy on certain time limit
  if ((t - start_eval_time) > EVALUATION_TIME)
  {
    // check if to stop the experiment. Negative value for MAX_EVALUATIONS will
    // never stop the experiment
    if (MAX_EVALUATIONS > 0 and generation_counter > MAX_EVALUATIONS)
    {
      std::cout << "#AccNEATLearner::update() Max Evaluations ("
                << MAX_EVALUATIONS << ") reached. stopping now." << std::endl;
      std::exit(0);
    }
    generation_counter++;
    //    std::cout
    //        << "#AccNEATLearner::update() EVALUATING NEW BRAIN (generation "
    //        << generation_counter
    //        << " )"
    //        << std::endl;

    double_t fitness = getFitness();
    std::cout << robot_name << " : " << generation_counter
              << ": fitness " << fitness << std::endl;
    this->writeCurrent(fitness);

    BaseController *new_controller = this->create_new_controller(fitness);
    if (new_controller not_eq active_controller.get())
    {
      this->active_controller.reset(new_controller);
    }
    start_eval_time = t;
    evaluator->start();
  }

  return BaseLearner::update(sensors, t, step);
}

BaseController *AccNEATLearner::create_new_controller(double fitness)
{
  if (current_evalaution)
  {
    // not first `create_new_controller`
    current_evalaution->finish(static_cast<float>(fitness));
  }
  current_evalaution = neat->Evaluation();
  NEAT::CpuNetwork *cppn = reinterpret_cast< NEAT::CpuNetwork * > (
          current_evalaution->Organism()->net.get());

  AccNEATCPPNController *controller =
          reinterpret_cast<AccNEATCPPNController *>(active_controller.get());
  controller->setCPPN(cppn);

  return controller;
}

float AccNEATLearner::getFitness()
{
  // Calculate fitness for current policy
  float fitness = static_cast<float>(evaluator->fitness());
  return fitness;
}

void AccNEATLearner::writeCurrent(double fitness)
{
  std::ofstream outputFile;
  outputFile.open(this->robot_name + ".log",
                  std::ios::app | std::ios::out | std::ios::ate);
  outputFile << "- generation: " << this->generation_counter << std::endl;
  outputFile << "  velocity: " << fitness << std::endl;
  // TODO: Should we record an entire generation?
  outputFile.close();
}
