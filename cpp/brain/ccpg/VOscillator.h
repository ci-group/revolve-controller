#ifndef REVOLVE_GAZEBO_BRAIN_VOSCILLATOR_H_
#define REVOLVE_GAZEBO_BRAIN_VOSCILLATOR_H_

#include "Neuron.h"
#include "NeuralConnection.h"


namespace revolve {
namespace brain {

class VOscillator : public Neuron
{
public:
	// VOscillator(sdf::ElementPtr neuron);
	VOscillator(const std::string &id,
		    const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double alpha_;
	double tau_;
	double energy_;

	double lastTime_;
	double stateDeriv_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_VOSCILLATOR_H_