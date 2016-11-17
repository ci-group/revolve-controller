#include "ExtendedNeuralNetwork.h"

#include <fstream>
#include <iostream>


namespace revolve {
namespace brain {

ExtendedNeuralNetwork::ExtendedNeuralNetwork(std::string modelName,
					     ExtNNConfig Config,
					     EvaluatorPtr evaluator,
					     const std::vector< ActuatorPtr > & actuators,
					     const std::vector< SensorPtr > & sensors)
{
	modelName_ = modelName;
	
	//evaluator_ = evaluator;
	inputs_ = Config.inputs_;
	outputs_ = Config.outputs_;
	allNeurons_ = Config.allNeurons_;
	inputNeurons_ = Config.inputNeurons_;
	outputNeurons_ = Config.outputNeurons_;
	hiddenNeurons_ = Config.hiddenNeurons_;
	outputPositionMap_ = Config.outputPositionMap_;
	inputPositionMap_ = Config.inputPositionMap_;
	idToNeuron_ = Config.idToNeuron_;
	connections_ = Config.connections_;
	numInputNeurons_ = Config.numInputNeurons_;
	numOutputNeurons_ = Config.numOutputNeurons_;
	numHiddenNeurons_ = Config.numHiddenNeurons_;
}


ExtendedNeuralNetwork::~ExtendedNeuralNetwork()
{
	delete [] inputs_;
	delete [] outputs_;
}


void ExtendedNeuralNetwork::connectionHelper(const std::string &src,
					     const std::string &dst,
					     const std::string &socket,
					     double weight,
					     const std::map<std::string, NeuronPtr> &idToNeuron)
{
	auto srcNeuron = idToNeuron.find(src);
	if (srcNeuron == idToNeuron.end()) {
		std::cerr << "Could not find source neuron '" << src << "'" << std::endl;
		throw std::runtime_error("Robot brain error");
	}
	auto dstNeuron = idToNeuron.find(dst);
	if (dstNeuron == idToNeuron.end()) {
		std::cerr << "Could not find destination neuron '" << dst << "'" << std::endl;
		throw std::runtime_error("Robot brain error");
	}

	NeuralConnectionPtr newConnection(new NeuralConnection(
		srcNeuron->second,
		dstNeuron->second,
		weight
	));

	// Add reference to this connection to the destination neuron
	(dstNeuron->second)->AddIncomingConnection(socket, newConnection);
	connections_.push_back(newConnection);
}




NeuronPtr ExtendedNeuralNetwork::addNeuron(const std::string &neuronId,
					   const std::string &neuronType,
					   const std::string &neuronLayer, // can be 'hidden', 'input' or 'output'
					   const std::map<std::string, double> &params)
{
	NeuronPtr newNeuron;

	if ("input" == neuronLayer) {
		newNeuron.reset(new InputNeuron(neuronId, params));

		this->inputNeurons_.push_back(newNeuron);
		inputPositionMap_[newNeuron] = numInputNeurons_;
		numInputNeurons_++;
	}

	else {

		if ("Sigmoid" == neuronType) {
			newNeuron.reset(new SigmoidNeuron(neuronId, params));
		}
		else if ("Simple" == neuronType) {
			newNeuron.reset(new LinearNeuron(neuronId, params));
		}
		else if ("Oscillator" == neuronType) {
			newNeuron.reset(new OscillatorNeuron(neuronId, params));
		}
		else if ("V-Neuron" == neuronType) {
			newNeuron.reset(new VOscillator(neuronId, params));
		}
		else if ("X-Neuron" == neuronType) {
			newNeuron.reset(new XOscillator(neuronId, params));
		}
		else if ("Bias" == neuronType) {
			newNeuron.reset(new BiasNeuron(neuronId, params));
		}
		else if ("Leaky" == neuronType) {
			newNeuron.reset(new LeakyIntegrator(neuronId, params));
		}
		else if ("DifferentialCPG" == neuronType) {
			newNeuron.reset(new DifferentialCPG(neuronId, params));
		}
		else {
			std::cerr << "Unsupported neuron type `" << neuronType << '`' << std::endl;
			throw std::runtime_error("Robot brain error");
		}

		if ("output" == neuronLayer) {
			this->outputNeurons_.push_back(newNeuron);
			outputPositionMap_[newNeuron] = numOutputNeurons_;
			numOutputNeurons_++;
		}
		else {
			this->hiddenNeurons_.push_back(newNeuron);
			numHiddenNeurons_++;
		}
	}

	this->allNeurons_.push_back(newNeuron);
	return newNeuron;
}


void ExtendedNeuralNetwork::update(const std::vector<ActuatorPtr>& actuators,
				   const std::vector<SensorPtr>& sensors,
				   double t,
				   double step) 
{
//   	std::cout << "yay! \n";
	//boost::mutex::scoped_lock lock(networkMutex_);

	// Read sensor data into the input buffer
	unsigned int p = 0;
	for (auto sensor : sensors) {
		sensor->read(&inputs_[p]);
		p += sensor->inputs();
	}

	// Feed inputs into the input neurons
	for (auto it = inputNeurons_.begin(); it != inputNeurons_.end(); ++it) {
		auto inNeuron = *it;
		int pos = inputPositionMap_[inNeuron];
		inNeuron->SetInput(inputs_[pos]);
	}

	// Calculate new states of all neurons
	for (auto it = allNeurons_.begin(); it != allNeurons_.end(); ++it) {
		(*it)->Update(t);
	}


	// Flip states of all neurons
	for (auto it = allNeurons_.begin(); it != allNeurons_.end(); ++it) {
		(*it)->FlipState();
	}

	// std::ofstream debF;
	// debF.open("/home/dmitry/projects/debug/debug_signals", std::ofstream::out | std::ofstream::app);
	for (auto it = outputNeurons_.begin(); it != outputNeurons_.end(); ++it) {
		auto outNeuron = *it;
		int pos = outputPositionMap_[outNeuron];
		outputs_[pos] = outNeuron->GetOutput();

		// debF << pos << "," << outputs_[pos] << std::endl;
	}

	// debF.close();
	
	// Send new signals to the actuators
	p = 0;
	for (auto actuator: actuators) {
		actuator->update(&outputs_[p], step);
		p += actuator->outputs();
	}
}

std::vector< double > ExtendedNeuralNetwork::GetWeights()
{
	std::vector<double> ret(connections_.size(), 0);
	for(unsigned int i = 0; i < connections_.size(); i++) {
		ret[i] = connections_[i]->GetWeight();
	}
	return ret;
}

void ExtendedNeuralNetwork::LoadWeights(std::vector< double > weights)
{
	if(weights.size() != connections_.size()) {
		std::cerr << "incorrect amount of weights (" << weights.size() << ") delivered. expected " << connections_.size() << std::endl;
		throw std::runtime_error("Weight size error");
	}
	for(unsigned int i = 0; i < weights.size(); i++) {
		connections_[i]->SetWeight(weights[i]);
	}
}



void ExtendedNeuralNetwork::flush()
{
	// Delete all references to incoming connections from neurons
	for (auto it = allNeurons_.begin(); it != allNeurons_.end(); ++it) {
		(*it)->DeleteIncomingConections();
	}

	// Nullify the input and output signal buffers
	for (int i = 0; i < numOutputNeurons_; ++i) {
		outputs_[i] = 0;
	}

	for (int i = 0; i < numInputNeurons_; ++i) {
		inputs_[i] = 0;
	}

	// Delete all references to connections from the list
	connections_.clear();

	// Delete all hidden neurons from the id->neuron map
	for (auto it = hiddenNeurons_.begin(); it != hiddenNeurons_.end(); ++it) {
		auto hiddenId = (*it)->Id();
		idToNeuron_.erase(hiddenId);
	}

	// Delete all hidden neurons
	// First delete all neurons
	// Then re-add input and output neurons

	allNeurons_.clear();
	hiddenNeurons_.clear();

	for (auto it = inputNeurons_.begin(); it != inputNeurons_.end(); ++it) {
		allNeurons_.push_back(*it);
	}
	for (auto it = outputNeurons_.begin(); it != outputNeurons_.end(); ++it) {
		allNeurons_.push_back(*it);
	}

    numHiddenNeurons_ = 0;
}

}
}
