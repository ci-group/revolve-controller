#ifndef REVOLVE_GAZEBO_BRAIN_DIFFERENTIALCPG_H_
#define REVOLVE_GAZEBO_BRAIN_DIFFERENTIALCPG_H_

#include "Neuron.h"
#include "NeuralConnection.h"

namespace revolve {
namespace brain {

class DifferentialCPG : public Neuron
{
public:

	DifferentialCPG(const std::string &id,
			const std::map<std::string, double> &params);
	
	virtual double CalculateOutput(double t);


protected:
	double bias_;
	double lastTime_;
};

}
}

#endif // REVOLVE_GAZEBO_BRAIN_DIFFERENTIALCPG_H_