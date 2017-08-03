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
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2016  Matteo De Carlo <matteo.dek@covolunablu.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef REVOLVEBRAIN_BRAIN_PYTHON_SUPGBRAIN_H_
#define REVOLVEBRAIN_BRAIN_PYTHON_SUPGBRAIN_H_

#include "brain/SUPGBrain.h"
#include <boost/python/list.hpp>
#include <vector>

namespace revolve {
namespace brain {

class SUPGBrain_python
        : public revolve::brain::SUPGBrain
{
public:
    SUPGBrain_python(revolve::brain::EvaluatorPtr evaluator,
                     const boost::python::list /*< boost::python::list< float > >*/ &neuron_coordinates,
                     const boost::python::list /*< ActuatorPtr> */ &motors,
                     const boost::python::list /*< SensorPtr> */ &sensors);


    void
    update(boost::python::list &actuators,
           const boost::python::list &sensors,
           double t,
           double step);

};

}
}

#endif // REVOLVEBRAIN_BRAIN_PYTHON_SUPGBRAIN_H_
