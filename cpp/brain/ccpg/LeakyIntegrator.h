#ifndef REVOLVE_GAZEBO_BRAIN_LEAKYINTEGRATOR_H_
#define REVOLVE_GAZEBO_BRAIN_LEAKYINTEGRATOR_H_

#include "Neuron.h"
#include "NeuralConnection.h"


namespace revolve {
namespace brain {

class LeakyIntegrator : public Neuron
{
public:
	LeakyIntegrator(const std::string &id,
			const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double bias_;
	double tau_;

	double lastTime_;

	double stateDeriv_;
	double state_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_LEAKYINTEGRATOR_H_