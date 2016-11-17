#ifndef REVOLVE_GAZEBO_BRAIN_NEURON_H_
#define REVOLVE_GAZEBO_BRAIN_NEURON_H_

//#include "Types.h"
#include "brain/brain.h"
#include <cstdlib>
#include <utility>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

//Types.h
namespace revolve {
namespace brain {
	class Motor;
	class VirtualSensor;
	class Brain;
	class MotorFactory;
	class SensorFactory;
	class ExtendedNeuralNetwork;
	
	class Neuron;
	class NeuralConnection;

	typedef boost::shared_ptr< Brain > BrainPtr;
	typedef boost::shared_ptr<ExtendedNeuralNetwork> NeuralNetworkPtr;
	typedef boost::shared_ptr< MotorFactory > MotorFactoryPtr;
	typedef boost::shared_ptr< SensorFactory > SensorFactoryPtr;

	typedef boost::shared_ptr<Neuron> NeuronPtr;
	typedef boost::shared_ptr<NeuralConnection> NeuralConnectionPtr;
}
}


namespace revolve {
namespace brain {

class Neuron
{
public:
	Neuron(const std::string &id);
	virtual ~Neuron() {};
	virtual double CalculateOutput(double t) = 0;

	void AddIncomingConnection(const std::string &socketName,
				   NeuralConnectionPtr connection);
	void DeleteIncomingConections();
	
	double GetOutput() const;

	virtual void SetInput(double value) {};

	void Update(double t);

	void FlipState();

	std::string GetSocketId() const;

	const std::string &Id() const;

protected:

	std::vector<std::pair<std::string, NeuralConnectionPtr> > incomingConnections_;
	double output_;
	double newOutput_;

	std::string id_;

};

}
}
#endif // REVOLVE_GAZEBO_BRAIN_NEURON_H_