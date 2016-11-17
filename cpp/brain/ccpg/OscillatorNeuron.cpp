#include "OscillatorNeuron.h"
#include <cmath>
#include <iostream>

namespace revolve {
namespace brain {

/* *
 * Constructor from a map of parameter names and values
 */
OscillatorNeuron::OscillatorNeuron(const std::string &id,
				   const std::map<std::string, double> &params):
Neuron(id)
{
	if (!params.count("rv:period") || !params.count("rv:phase_offset") || !params.count("rv:amplitude")) {
		std::cerr << "An `" << "Oscillator" << 
		"` neuron requires `rv:period`, `rv:phase_offset` and `rv:amplitude` elements." << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	
	this->period_ = params.find("rv:period")->second;
	this->phaseOffset_ = params.find("rv:phase_offset")->second;
	this->gain_ = params.find("rv:amplitude")->second;
}


double OscillatorNeuron::CalculateOutput(double t)
{
	return 0.5 * (1.0 + this->gain_ *
		sin( 2.0*M_PI/(this->period_) * (t - this->period_ * this->phaseOffset_) ));
}


}
}