#include "XOscillator.h"
#include <iostream>

namespace revolve {
namespace brain {


XOscillator::XOscillator(const std::string &id,
			 const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:tau")) {
		std::cerr << "A `" << "X-Oscillator" << 
		"` neuron requires an `rv:tau` element." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	
	this->tau_ = params.find("rv:tau")->second;

	this->lastTime_ = 0;
	this->stateDeriv_ = 0;
}


double XOscillator::CalculateOutput(double t)
{
	double deltaT = t - lastTime_;
	lastTime_ = t;

	if (deltaT > 0.1) {
		deltaT = 0.1;
	}

	double vInput = 0;  // input from V-neuron of the same oscillator
	double xInput = this->output_; // input from X-neuron of the same oscillator (this neuron)

	for (auto it = this->incomingConnections_.begin(); it != this->incomingConnections_.end(); ++it)
	{
		auto socketId = it->first;
		auto inConnection = it->second;

		if (socketId == "from_v") {
			vInput += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
		}
	}

	stateDeriv_ = vInput / tau_;



	double result = xInput + deltaT * stateDeriv_;
	if (result > 1000.0) {
		result = 1000.0;
	}
	if (result < -1000.0) {
		result = -1000.0;
	}

	return result;
}


}
}