#ifndef REVOLVE_GAZEBO_BRAIN_XOSCILLATOR_H_
#define REVOLVE_GAZEBO_BRAIN_XOSCILLATOR_H_

#include "Neuron.h"
#include "NeuralConnection.h"

namespace revolve {
namespace brain {

class XOscillator : public Neuron
{
public:
	// XOscillator(sdf::ElementPtr neuron);
	XOscillator(const std::string &id,
		    const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double tau_;

	double lastTime_;
	double stateDeriv_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_XOSCILLATOR_H_