#include "LinearNeuron.h"
#include <sstream>
#include <iostream>

namespace revolve {
namespace brain {


/* *
 * Constructor from a map of parameter names and values
 */
LinearNeuron::LinearNeuron(const std::string &id,
			   const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:bias") || !params.count("rv:gain")) {
		std::cerr << "A `" << "Simple" << "` neuron requires `rv:bias` and `rv:gain` elements." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	this->bias_ = params.find("rv:bias")->second;
	this->gain_ = params.find("rv:gain")->second;
}


double LinearNeuron::CalculateOutput(double t)
{
	double inputValue = 0;

	for (auto it = this->incomingConnections_.begin(); it != this->incomingConnections_.end(); ++it)
	{
		auto inConnection = it->second;
		inputValue += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
	}

	double result = this->gain_ * (inputValue - this->bias_);

	return result;
}


}
}