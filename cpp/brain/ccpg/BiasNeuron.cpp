#include "BiasNeuron.h"
#include <iostream>

namespace revolve {
namespace brain {

/* *
 * Constructor from a map of parameter names and values
 */
BiasNeuron::BiasNeuron(const std::string &id,
		       const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:bias")) {
		std::cerr << "A `" << "Bias" << "` neuron requires `rv:bias` element." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	this->bias_ = params.find("rv:bias")->second;
}


double BiasNeuron::CalculateOutput(double /*t*/)
{
	return this->bias_;
}


}
}