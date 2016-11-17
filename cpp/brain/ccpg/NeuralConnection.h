#ifndef REVOLVE_GAZEBO_BRAIN_NEURALCONNECTION_H_
#define REVOLVE_GAZEBO_BRAIN_NEURALCONNECTION_H_

#include "Neuron.h"

namespace revolve {
namespace brain {

class NeuralConnection
{
public:
	NeuralConnection(const NeuronPtr &src,
			 const NeuronPtr &dst,
			 double weight);
	double GetWeight() const;
	void SetWeight(double weigth);
	NeuronPtr GetInputNeuron() const;

protected:
	double weight_;
	NeuronPtr src_;
	NeuronPtr dst_;
};


}
}

#endif // REVOLVE_GAZEBO_BRAIN_NEURALCONNECTION_H_