#include "mutator.h"

#include <algorithm>
#include <iostream>

namespace CPPNEAT {

std::vector<Neuron::Ntype> Mutator::get_addable_types(std::map<Neuron::Ntype, Neuron::NeuronTypeSpec> brain_spec) {
	std::vector<Neuron::Ntype> possible;
	for(auto it : brain_spec) {
		Neuron::NeuronTypeSpec spec = it.second;
		if(std::find(spec.possible_layers.begin(), spec.possible_layers.end(), Neuron::HIDDEN_LAYER) != spec.possible_layers.end()) {
			possible.push_back(it.first);
		}
	}
	return possible;
}

Mutator::Mutator(std::map<Neuron::Ntype, Neuron::NeuronTypeSpec> brain_spec,
		 double new_connection_sigma,
		 int innovation_number,
		 int max_attempts,
		 std::vector<Neuron::Ntype> addable_neurons,
		 bool layered) 
	: brain_spec(brain_spec)
	, new_connection_sigma(new_connection_sigma)
	, innovation_number(innovation_number)
	, max_attempts(max_attempts)
	, addable_neurons(addable_neurons)
	, layered(layered) {
	std::random_device rd;
	generator.seed(rd());
	if(addable_neurons.size() == 0) {
		this->addable_neurons = get_addable_types(brain_spec);
	}
	
}
void Mutator::make_starting_genotype_known(GeneticEncodingPtr genotype) {
	for(ConnectionGenePtr connection_gene : genotype->connection_genes) {
		std::pair<int,int> connection_innovation(connection_gene->mark_from, connection_gene->mark_to);
		connection_innovations[connection_innovation] = connection_gene->getInnovNumber();
	}
}

void Mutator::mutate_neuron_params(GeneticEncodingPtr genotype, double probability, double sigma) {
	std::uniform_real_distribution<double> uniform(0,1);
	if(!genotype->layered) {
		for(NeuronGenePtr neuron_gene : genotype->neuron_genes) {
			if(uniform(generator) < probability) {
				std::vector<Neuron::ParamSpec> neuron_params = brain_spec[neuron_gene->neuron->neuron_type].param_specs;
				if(neuron_params.size() > 0) {
					std::uniform_int_distribution<int> uniform_int(0, neuron_params.size()-1);
					Neuron::ParamSpec param = neuron_params[uniform_int(generator)];
					double cur_val = neuron_gene->neuron->neuron_params[param.name];
// 					std::normal_distribution<double> normal(0,sigma*(param.max_value-param.min_value));
					std::normal_distribution<double> normal(0,sigma);
					cur_val += normal(generator);
					neuron_gene->neuron->set_neuron_param(cur_val, param);				
				}
			}
		}
	} else {
		for(std::vector<NeuronGenePtr> layer : genotype->layers) {
			for(NeuronGenePtr neuron_gene : layer) {
				if(uniform(generator) < probability) {
					std::vector<Neuron::ParamSpec> neuron_params = brain_spec[neuron_gene->neuron->neuron_type].param_specs;
					if(neuron_params.size() > 0) {
						std::uniform_int_distribution<int> uniform_int(0, neuron_params.size()-1);
						Neuron::ParamSpec param = neuron_params[uniform_int(generator)];
						double cur_val = neuron_gene->neuron->neuron_params[param.name];
// 						std::normal_distribution<double> normal(0,sigma*(param.max_value-param.min_value));
						std::normal_distribution<double> normal(0,sigma);
						cur_val += normal(generator);
						neuron_gene->neuron->set_neuron_param(cur_val, param);				
					}
				}
			}
		}
	}
}

void Mutator::mutate_weights(GeneticEncodingPtr genotype, double probability, double sigma) {
	std::uniform_real_distribution<double> uniform(0,1);
	std::normal_distribution<double> normal(0, sigma);
	for(ConnectionGenePtr connection_gene : genotype->connection_genes) {
		if(uniform(generator) < probability) {
			connection_gene->weight += normal(generator);
		}
	}
}

void Mutator::mutate_structure(GeneticEncodingPtr genotype, double probability) {
	std::uniform_real_distribution<double> uniform(0,1);
	if(uniform(generator) < probability) {
		if(genotype->connection_genes.size() == 0) {
			add_connection_mutation(genotype, new_connection_sigma);
		} else {
			if(uniform(generator) < 0.5) {
				add_connection_mutation(genotype, new_connection_sigma);
			} else {
				add_neuron_mutation(genotype, new_connection_sigma);
			}
		}
	}
}

bool Mutator::add_connection_mutation(GeneticEncodingPtr genotype, double sigma) {
	if(!genotype->layered) {
		std::uniform_int_distribution<int> choice(0,genotype->neuron_genes.size() -1);
		NeuronGenePtr neuron_from = genotype->neuron_genes[choice(generator)];
		NeuronGenePtr neuron_to = genotype->neuron_genes[choice(generator)];
		int mark_from = neuron_from->getInnovNumber();
		int mark_to = neuron_to->getInnovNumber();
		
		int num_attempts = 1;
		
		while(genotype->connection_exists(mark_from, mark_to) || neuron_to->neuron->layer == Neuron::INPUT_LAYER) {
			neuron_from = genotype->neuron_genes[choice(generator)];
			neuron_to = genotype->neuron_genes[choice(generator)];
			mark_from = neuron_from->getInnovNumber();
			mark_to = neuron_to->getInnovNumber();
			
			num_attempts++;
			if(num_attempts > max_attempts) {
				return false;
			}
		}
		std::normal_distribution<double> normal(0,sigma);
		add_connection(mark_from, mark_to, normal(generator), genotype, "");
#ifdef CPPNEAT_DEBUG
		if(!genotype->is_valid()) {
			std::cerr << "add connection mutation caused invalid genotye" << std::endl;
			throw std::runtime_error("mutation error");
		}
#endif
		return true;
	} else {
		std::uniform_int_distribution<unsigned int> choice(0,genotype->num_neuron_genes() -1);
		std::pair<unsigned int, unsigned int> index_from = genotype->convert_index_to_layer_index(choice(generator));
		std::pair<unsigned int, unsigned int> index_to = genotype->convert_index_to_layer_index(choice(generator));
		NeuronGenePtr neuron_from = genotype->layers[index_from.first][index_from.second];
		NeuronGenePtr neuron_to = genotype->layers[index_to.first][index_to.second];
		int mark_from = neuron_from->getInnovNumber();
		int mark_to = neuron_to->getInnovNumber();
		
		int num_attempts = 1;
		
		while(genotype->connection_exists(mark_from, mark_to) || index_from.first >= index_to.first) {
			index_from = genotype->convert_index_to_layer_index(choice(generator));
			index_to = genotype->convert_index_to_layer_index(choice(generator));
			neuron_from = genotype->layers[index_from.first][index_from.second];
			neuron_to = genotype->layers[index_to.first][index_to.second];
			mark_from = neuron_from->getInnovNumber();
			mark_to = neuron_to->getInnovNumber();
			
			num_attempts++;
			if(num_attempts > max_attempts) {
				return false;
			}
		}
		std::normal_distribution<double> normal(0,sigma);
		add_connection(mark_from, mark_to, normal(generator), genotype, "");
#ifdef CPPNEAT_DEBUG
		if(!genotype->is_valid()) {
			std::cerr << "add connection mutation caused invalid genotye" << std::endl;
			throw std::runtime_error("mutation error");
		}
#endif
		return true;
	}
}

std::map<std::string, double> get_random_parameters(Neuron::NeuronTypeSpec param_specs, double sigma) {
	std::map<std::string, double> params;
	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normal(0,sigma);
	for(Neuron::ParamSpec spec : param_specs.param_specs) {
// 		params[spec.name] = spec.min_value + normal(generator)*(spec.max_value - spec.min_value);
// 		if(!spec.min_inclusive) {
// 			params[spec.name] = std::max(params[spec.name], spec.min_value + spec.epsilon);
// 		}
// 		if(!spec.max_inclusive) {
// 			params[spec.name] = std::min(params[spec.name], spec.max_value - spec.epsilon);
// 		}
		params[spec.name] = normal(generator);

	}
	return params;
}
void Mutator::add_neuron_mutation(GeneticEncodingPtr genotype, double sigma) {
	assert(genotype->connection_genes.size() > 0);
	assert(addable_neurons.size() > 0);
	if(!genotype->layered) {
		std::uniform_int_distribution<int> choice1(0,genotype->connection_genes.size() -1);
		int split_id = choice1(generator);
		ConnectionGenePtr split = genotype->connection_genes[split_id];
		
		double old_weight = split->weight;
		int mark_from = split->mark_from;
		int mark_to = split->mark_to;
		
		genotype->remove_connection_gene(split_id);
		NeuronPtr neuron_from = boost::dynamic_pointer_cast<NeuronGene>(genotype->find_gene_by_in(mark_from))->neuron;
		NeuronPtr neuron_to = boost::dynamic_pointer_cast<NeuronGene>(genotype->find_gene_by_in(mark_to))->neuron;
		
		std::uniform_int_distribution<int> choice2(0,addable_neurons.size()-1);
		Neuron::Ntype new_neuron_type = addable_neurons[choice2(generator)];
		
		std::map<std::string, double> new_neuron_params = get_random_parameters(brain_spec[new_neuron_type], sigma);
		
		NeuronPtr neuron_middle(new Neuron("augment " + std::to_string(innovation_number+1), 
						Neuron::HIDDEN_LAYER,
						new_neuron_type,
						new_neuron_params));
		int mark_middle = add_neuron(neuron_middle, genotype, split);
		add_connection(mark_from, mark_middle, old_weight, genotype, "");
		add_connection(mark_middle, mark_to, 1.0, genotype, "");
	} else {
		std::uniform_int_distribution<int> choice1(0,genotype->connection_genes.size() -1);
		int split_id = choice1(generator);
		ConnectionGenePtr split = genotype->connection_genes[split_id];
		
		double old_weight = split->weight;
		int mark_from = split->mark_from;
		int mark_to = split->mark_to;
		
		genotype->remove_connection_gene(split_id);
		NeuronPtr neuron_from = boost::dynamic_pointer_cast<NeuronGene>(genotype->find_gene_by_in(mark_from))->neuron;
		NeuronPtr neuron_to = boost::dynamic_pointer_cast<NeuronGene>(genotype->find_gene_by_in(mark_to))->neuron;
		
		std::uniform_int_distribution<int> choice2(0,addable_neurons.size()-1);
		Neuron::Ntype new_neuron_type = addable_neurons[choice2(generator)];
		
		std::map<std::string, double> new_neuron_params = get_random_parameters(brain_spec[new_neuron_type], sigma);
		
		NeuronPtr neuron_middle(new Neuron("augment " + std::to_string(innovation_number+1), 
						Neuron::HIDDEN_LAYER,
						new_neuron_type,
						new_neuron_params));
		int mark_middle = add_neuron(neuron_middle, genotype, split);
#ifdef CPPNEAT_DEBUG
		if(!genotype->is_valid()) {
			std::cerr << "add neuron mutation caused invalid genotye1" << std::endl;
			throw std::runtime_error("mutation error");
		}
#endif
		add_connection(mark_from, mark_middle, old_weight, genotype, "");
#ifdef CPPNEAT_DEBUG
		if(!genotype->is_valid()) {
			std::cerr << "add neuron mutation caused invalid genotye2" << std::endl;
			throw std::runtime_error("mutation error");
		}
#endif
		add_connection(mark_middle, mark_to, 1.0, genotype, "");
	}
#ifdef CPPNEAT_DEBUG
	if(!genotype->is_valid()) {
		std::cerr << "add neuron mutation caused invalid genotye" << std::endl;
		throw std::runtime_error("mutation error");
	}
#endif
}

void Mutator::remove_connection_mutation(GeneticEncodingPtr genotype) {
	if(genotype->connection_genes.size() == 0) {
		return;
	}
	std::uniform_int_distribution<int> choice(0, genotype->connection_genes.size()-1);
	genotype->connection_genes.erase(genotype->connection_genes.begin() + choice(generator));
}

void Mutator::remove_neuron_mutation(GeneticEncodingPtr genotype) {
	std::vector<int> hidden_neuron_ids;
	for(unsigned int i = 0; i < genotype->neuron_genes.size(); i++) {
		if(genotype->neuron_genes[i]->neuron->layer == Neuron::HIDDEN_LAYER) {
			hidden_neuron_ids.push_back(i);
		}
	}
	std::uniform_int_distribution<int> choice(0, hidden_neuron_ids.size() - 1);
	int gene_id = hidden_neuron_ids[choice(generator)];
	NeuronGenePtr neuron_gene = genotype->neuron_genes[gene_id];
	int neuron_mark = neuron_gene->getInnovNumber();
	
	std::vector<int> bad_connections;
	for(unsigned int i = 0; i < genotype->connection_genes.size(); i++) {
		if(genotype->connection_genes[i]->mark_from == neuron_mark || genotype->connection_genes[i]->mark_to == neuron_mark) {
			bad_connections.push_back(i);
		}
	}
	for(int i = bad_connections.size() -1 ; i >= 0; i--) {
		genotype->remove_connection_gene(bad_connections[i]);
	}
	genotype->remonve_neuron_gene(gene_id);
}
int Mutator::add_neuron(NeuronPtr neuron, GeneticEncodingPtr genotype, ConnectionGenePtr split) {
	int connection_split_in = split->getInnovNumber();
	std::pair<int, Neuron::Ntype> neuron_pair(connection_split_in, neuron->neuron_type);	
	if(neuron_innovations.find(neuron_pair) != neuron_innovations.end()) {
		unsigned int i = 0;
		while(i < neuron_innovations[neuron_pair].size()
		      && genotype->find_gene_by_in(neuron_innovations[neuron_pair][i]) != nullptr) {
			i++;
		}
		//some previous innovation is not are already present in the genome-> add a it
		if(i < neuron_innovations[neuron_pair].size())
		{
			NeuronGenePtr new_neuron_gene(new NeuronGene(neuron, neuron_innovations[neuron_pair][i], true));
			if(!genotype->layered) {
				genotype->add_neuron_gene(new_neuron_gene);
			} else {
				int mark_from = split->mark_from;
				int mark_to = split->mark_to;
				std::pair<unsigned int, unsigned int> index_from = genotype->convert_in_to_layer_index(mark_from);
				std::pair<unsigned int, unsigned int> index_to = genotype->convert_in_to_layer_index(mark_to);
				assert(index_from.first < index_to.first);
				genotype->add_neuron_gene(new_neuron_gene, 
							  index_from.first + 1,
							  index_from.first + 1 == index_to.first); //we need a new layer iff the two layers are "right next to each other"
				
			}
			return new_neuron_gene->getInnovNumber();
		} 
	}
	//new innovation -> add new neuron with new innovation number
	NeuronGenePtr new_neuron_gene(new NeuronGene(neuron, ++innovation_number, true));
	//in base case a new vector is constructed here
	neuron_innovations[neuron_pair].push_back(innovation_number);
	if(!genotype->layered) {
		genotype->add_neuron_gene(new_neuron_gene);
	} else {
		int mark_from = split->mark_from;
		int mark_to = split->mark_to;
		std::pair<unsigned int, unsigned int> index_from = genotype->convert_in_to_layer_index(mark_from);
		std::pair<unsigned int, unsigned int> index_to = genotype->convert_in_to_layer_index(mark_to);
		assert(index_from.first < index_to.first);
		genotype->add_neuron_gene(new_neuron_gene, 
						index_from.first + 1,
						index_from.first + 1 == index_to.first); //we need a new layer iff the two layers are "right next to each other"
		
	}
	return new_neuron_gene->getInnovNumber();
}

int Mutator::add_connection(int mark_from, int mark_to, double weight, GeneticEncodingPtr genotype, std::string socket) {
	std::pair<int,int> innovation_pair(mark_from, mark_to);
	if(connection_innovations.find(innovation_pair) != connection_innovations.end()) {
		GenePtr found = genotype->find_gene_by_in(connection_innovations[innovation_pair]);
		if(found != nullptr) {
			boost::dynamic_pointer_cast<ConnectionGene>(found)->setEnabled(true);
			return connection_innovations[innovation_pair];
		} else {
			ConnectionGenePtr new_conn_gene(new ConnectionGene(mark_to,
									   mark_from,
									   weight,
									   connection_innovations[innovation_pair],
									   true,
									   socket));
			genotype->add_connection_gene(new_conn_gene);
			return new_conn_gene->getInnovNumber();
		}
	} 
	ConnectionGenePtr new_conn_gene(new ConnectionGene(mark_to,
							   mark_from,
							   weight,
							   ++innovation_number,
							   true,
						           socket));
	connection_innovations[innovation_pair] = innovation_number;
	genotype->add_connection_gene(new_conn_gene);
	return new_conn_gene->getInnovNumber();
}
//old
// int Mutator::add_neuron(NeuronPtr neuron, GeneticEncodingPtr genotype) {
// 	NeuronGenePtr new_neuron_gene(new NeuronGene(neuron, ++innovation_number, true));
// 	genotype->add_neuron_gene(new_neuron_gene);
// 	return new_neuron_gene->getInnovNumber();
// }
// 
// int Mutator::add_connection(int mark_from, int mark_to, double weight, GeneticEncodingPtr genotype, std::string socket) {
// 	ConnectionGenePtr new_conn_gene(new ConnectionGene(mark_to,
// 							   mark_from,
// 							   weight,
// 							   ++innovation_number,
// 							   true,
// 						           socket));
// 	genotype->add_connection_gene(new_conn_gene);
// 	return new_conn_gene->getInnovNumber();
// }
	
}