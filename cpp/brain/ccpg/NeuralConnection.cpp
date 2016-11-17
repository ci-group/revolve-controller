#include "NeuralConnection.h"

namespace revolve {
namespace brain {

NeuralConnection::NeuralConnection(const NeuronPtr &src,
				   const NeuronPtr &dst,
				   double weight)
{
	weight_ = weight;
	src_ = src;
	dst_ = dst;
}

double NeuralConnection::GetWeight() const
{
	return weight_;
}

void NeuralConnection::SetWeight(double weight) 
{
	weight_ = weight;
}
NeuronPtr NeuralConnection::GetInputNeuron() const
{
	return src_;
}

}
}