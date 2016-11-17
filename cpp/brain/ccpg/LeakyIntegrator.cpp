#include "LeakyIntegrator.h"
#include <cmath>
#include <iostream>

namespace revolve {
namespace brain {

LeakyIntegrator::LeakyIntegrator(const std::string &id,
				 const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:bias") || !params.count("rv:tau")) {
		std::cerr << "A `" << "Leaky Integrator" << 
		"` neuron requires `rv:bias` and `rv:tau` elements." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	
	this->bias_ = params.find("rv:bias")->second;
	this->tau_ = params.find("rv:tau")->second;

	this->lastTime_ = 0;
	this->stateDeriv_ = 0;
	this->state_ = 0;
}


double LeakyIntegrator::CalculateOutput(double t)
{
	double deltaT = t - lastTime_;
	lastTime_ = t;

	if (deltaT > 0.1) {
		deltaT = 0.1;
	}

	double inputValue = 0;

	// Calculate the input value
	for (auto it = this->incomingConnections_.begin(); it != this->incomingConnections_.end(); ++it)
	{
		auto inConnection = it->second;
		inputValue += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
	}

	stateDeriv_ = ( -state_ + inputValue ) / tau_;
	state_ = state_ + deltaT * stateDeriv_;

	double result = 1.0 / (1.0 + exp(state_ + bias_));

	return result;
}


}
}