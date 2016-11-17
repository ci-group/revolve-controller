#ifndef REVOLVE_GAZEBO_BRAIN_INPUTNEURON_H_
#define REVOLVE_GAZEBO_BRAIN_INPUTNEURON_H_

#include "Neuron.h"

namespace revolve {
namespace brain {

class InputNeuron : public Neuron
{
public:
	// InputNeuron(sdf::ElementPtr neuron);
	InputNeuron(const std::string &id,
		    const std::map<std::string, double> &params);

	virtual void SetInput(double value);
	virtual double CalculateOutput(double t);

protected:
	double input_;

};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_INPUTNEURON_H_