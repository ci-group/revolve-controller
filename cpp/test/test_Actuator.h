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
*
*/

#ifndef TESTACTUATOR_H
#define TESTACTUATOR_H

#include "brain/Actuator.h"

class TestActuator : public revolve::brain::Actuator
{
public:
    TestActuator(bool verbose = false);

    virtual void update(double *output_vector,
                        double step) override;
    virtual unsigned int outputs() const override;

private:
    bool verbose;
};

#endif // TESTACTUATOR_H
