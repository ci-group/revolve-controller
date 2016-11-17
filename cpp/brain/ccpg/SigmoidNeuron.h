#ifndef REVOLVE_GAZEBO_BRAIN_SIGMOIDNEURON_H_
#define REVOLVE_GAZEBO_BRAIN_SIGMOIDNEURON_H_

#include "Neuron.h"
#include "NeuralConnection.h"

namespace revolve {
namespace brain {

class SigmoidNeuron : public Neuron
{
public:
	// SigmoidNeuron(sdf::ElementPtr neuron);
	SigmoidNeuron(const std::string &id,
		      const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double gain_;
	double bias_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_SIGMOIDNEURON_H_