#ifndef REVOLVE_GAZEBO_BRAIN_OSCILLATORNEURON_H_
#define REVOLVE_GAZEBO_BRAIN_OSCILLATORNEURON_H_

#include "Neuron.h"
#include "NeuralConnection.h"

namespace revolve {
namespace brain {

class OscillatorNeuron : public Neuron
{
public:
	// OscillatorNeuron(sdf::ElementPtr neuron);
	OscillatorNeuron(const std::string &id,
			 const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);

protected:
	double period_;
	double phaseOffset_;
	double gain_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_OSCILLATORNEURON_H_