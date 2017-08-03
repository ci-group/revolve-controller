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

#include "LinearNeuron.h"
#include <sstream>
#include <iostream>

namespace revolve {
namespace brain {


LinearNeuron::LinearNeuron(const std::string &id,
                           const std::map<std::string, double> &params) :
        Neuron(id)
{
  if (not params.count("rv:bias") || not params.count("rv:gain")) {
    std::cerr << "A `" << "Simple" << "` neuron requires `rv:bias` and `rv:gain` elements." << std::endl;
    throw std::runtime_error("Robot brain error");
  }
  this->bias_ = params.find("rv:bias")->second;
  this->gain_ = params.find("rv:gain")->second;
}


double
LinearNeuron::CalculateOutput(double t)
{
  double inputValue = 0;

  for (auto it = this->incomingConnections_.begin(); it != this->incomingConnections_.end(); ++it) {
    auto inConnection = it->second;
    inputValue += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
  }

  double result = this->gain_ * (inputValue - this->bias_);

  return result;
}

std::map<std::string, double>
LinearNeuron::getNeuronParameters()
{
  std::map<std::string, double> ret;
  ret["rv:bias"] = bias_;
  ret["rv:gain"] = gain_;
  return ret;
}

void
LinearNeuron::setNeuronParameters(std::map<std::string, double> params)
{
  if (not params.count("rv:bias") || not params.count("rv:gain")) {
    std::cerr << "A `" << "Simple" << "` neuron requires `rv:bias` and `rv:gain` elements." << std::endl;
    throw std::runtime_error("Robot brain error");
  }
  this->bias_ = params.find("rv:bias")->second;
  this->gain_ = params.find("rv:gain")->second;
}


std::string
LinearNeuron::getType()
{
  return "Simple";
}


}
}
