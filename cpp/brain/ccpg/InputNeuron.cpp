#include "InputNeuron.h"

namespace revolve {
namespace brain {

// InputNeuron::InputNeuron(sdf::ElementPtr neuron)
// {
// 	input_ = 0;
// }


InputNeuron::InputNeuron(const std::string &id,
			 const std::map<std::string, double> &params):
Neuron(id)
{
	input_ = 0;
}

double InputNeuron::CalculateOutput(double /*t*/)
{
	return input_;
}

void InputNeuron::SetInput(double value)
{
	input_ = value;
}

}
}