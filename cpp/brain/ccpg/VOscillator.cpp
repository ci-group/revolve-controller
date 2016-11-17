#include "VOscillator.h"
#include <cmath>
#include <iostream>


namespace revolve {
namespace brain {


VOscillator::VOscillator(const std::string &id,
			 const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:alpha") || !params.count("rv:tau") || !params.count("rv:energy")) {
		std::cerr << "A `" << "V-Oscillator" << 
		"` neuron requires `rv:alpha`, `rv:tau` and `rv:energy` elements." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	
	this->alpha_ = params.find("rv:alpha")->second;
	this->tau_ = params.find("rv:tau")->second;
	this->energy_ = params.find("rv:energy")->second;

	this->lastTime_ = 0;
	this->stateDeriv_ = 0;
	this->output_ = sqrt(this->energy_);
}


double VOscillator::CalculateOutput(double t)
{
	double deltaT = t - lastTime_;
	lastTime_ = t;

	if (deltaT > 0.1) {
		deltaT = 0.1;
	}

	double xInput = 0;  // input from X-neuron of the same oscillator
	double vInput = this->output_; // input from V-neuron of the same oscillator (this neuron)

	double vExternal = 0; // input from V-neuron of another oscillator
	double xExternal = 0; // input from X-neuron of another oscillator

	double otherInputs = 0; // all other inputs

	for (auto it = this->incomingConnections_.begin(); it != this->incomingConnections_.end(); ++it)
	{
		auto socketId = it->first;
		auto inConnection = it->second;

		

		if (socketId == "from_x") {
			xInput += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
		}
		else if (socketId == "from_x_ext") {
			xExternal += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
		} 
		else if (socketId == "from_v_ext") {
			vExternal += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
		}

		else {
			otherInputs += inConnection->GetInputNeuron()->GetOutput() * inConnection->GetWeight();
		}

		

	}

	stateDeriv_ = (- (alpha_ / energy_) * vInput * ( xInput*xInput + vInput*vInput )
		+ alpha_ * vInput - xInput + xExternal + vExternal + otherInputs) / tau_;

	double result = vInput + deltaT * stateDeriv_;

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