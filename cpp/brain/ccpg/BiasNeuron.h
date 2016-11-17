#ifndef REVOLVE_GAZEBO_BRAIN_BIASNEURON_H_
#define REVOLVE_GAZEBO_BRAIN_BIASNEURON_H_

#include "Neuron.h"
#include "NeuralConnection.h"

namespace revolve {
namespace brain {

class BiasNeuron : public Neuron
{
public:
	BiasNeuron(const std::string &id,
		   const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double gain_;
	double bias_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_BIASNEURON_H_