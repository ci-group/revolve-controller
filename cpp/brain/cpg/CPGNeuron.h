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
* Date: November 1, 2016
*
*/

#ifndef REVOLVE_BRAIN_CPGNEURON_H
#define REVOLVE_BRAIN_CPGNEURON_H

#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <exception>

#include "CPGTypes.h"

namespace revolve
{
  namespace brain
  {
    namespace cpg
    {
      class Neuron
      {
        public:
        /// \brief Virtual function to standardize all neurons
        /// \param inputs std::vector<real> containing the neuron inputs
        /// \param delta_time time passed since the last update. Value in
        /// seconds expected to be positive and relativly small. Big time steps
        /// could have weird effects on the network.
        /// \throws should throw invalid_input_exception if input vector is not
        /// of the correct size
        /// \return std::vector<real> containing the network output
        virtual std::vector< real_t > update(
                std::vector< real_t > inputs,
                real_t delta_time) = 0;

        /// \brief Exception for update function, in case the input vector is
        /// not ok for the neuron
        class invalid_input_exception
                : public std::exception
        {
          public:
          invalid_input_exception(std::string desc)
                  : std::exception()
                  , desc(desc)
          {}

          virtual const char *what() const throw()
          {
            return desc.c_str();
          }

          private:
          const std::string desc;
        };

        /// \brief  Exception for parameters setters and constructors in Neurons
        class invalid_parameter
                : public std::exception
        {
          public:
          invalid_parameter(
                  std::string param_name,
                  real_t value,
                  real_t min,
                  real_t max)
                  : param_name(param_name)
                  , value(value)
                  , min(min)
                  , max(max)
          {}

          virtual const char *what() const throw()
          {
            std::stringstream desc;
            desc << "Invalid value " << value << " for parameter "
                 << param_name
                 << ". It should be between "
                 << min << " and " << max;
            return desc.str().c_str();
          }

          private:
          const std::string param_name;
          const real_t value;
          const real_t min;
          const real_t max;
        };

        protected:
        static real_t percentage_from_range(
                real_t percentage,
                real_t range_start,
                real_t range_end)
        {
          if (percentage > 1)
          {
            return range_end;
          }
          else if (percentage < 0)
          {
            return range_start;
          }

          real_t range = range_end - range_start;
          real_t distance = percentage * range;
          return range_start + distance;
        }

        static real_t percentage_of_range(
                real_t value,
                real_t range_start,
                real_t range_end)
        {
          if (value > range_end)
          {
            return 1;
          }
          else if (value < range_start)
          {
            return 0;
          }

          real_t range = range_end - range_start;
          value = value - range_start;
          return value / range;
        }
      };
    }
  }
}

#endif  // REVOLVE_BRAIN_CPGNEURON_H
