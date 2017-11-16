#include "CPPNMutator.h"

#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

namespace CPPNEAT
{
  /////////////////////////////////////////////////
  std::vector< Neuron::Ntype > Mutator::AddableTypes(
          std::map< Neuron::Ntype, Neuron::NeuronTypeSpec > _specifications)
  {
    std::vector< Neuron::Ntype > possibleTypes;
    for (const auto &specificationPair : _specifications)
    {
      Neuron::NeuronTypeSpec specification = specificationPair.second;
      auto layer = std::find(
              specification.possible_layers.begin(),
              specification.possible_layers.end(),
              Neuron::HIDDEN_LAYER);
      if (layer not_eq specification.possible_layers.end())
      {
        possibleTypes.push_back(specificationPair.first);
      }
    }
    return possibleTypes;
  }

  /////////////////////////////////////////////////
  Mutator::Mutator(
          std::map< Neuron::Ntype, Neuron::NeuronTypeSpec > _specification,
          const double _connectionSigma,
          const size_t _innovationNumber,
          const size_t _maxAttempts,
          std::vector< Neuron::Ntype > _availableNeurons)
          : specification_(_specification)
          , sigma_(_connectionSigma)
          , innovationNumber_(_innovationNumber)
          , maxAttempts_(_maxAttempts)
          , availableNeurons_(_availableNeurons)
  {
    std::random_device rd;
    generator_.seed(rd());
    if (_availableNeurons.empty())
    {
      this->availableNeurons_ = this->AddableTypes(_specification);
    }

  }

  /////////////////////////////////////////////////
  void Mutator::make_starting_genotype_known(GeneticEncodingPtr _genotype)
  {
    for (const auto &connection : _genotype->connections_)
    {
      this->connectionInnovations_[{connection->From(), connection->To()}] =
              connection->InnovationNumber();
    }
  }

  /////////////////////////////////////////////////
  void Mutator::RecordInnovations(const std::string &_path)
  {
    std::ofstream outputFile;
    outputFile.open(_path,
                    std::ios::out | std::ios::trunc);
    outputFile << "- in_no: " << innovationNumber_ << std::endl;
    outputFile << "- connection_innovations: " << std::endl;
    for (const auto &connection : connectionInnovations_)
    {
      outputFile << "  - connection_innovation: " << std::endl;
      outputFile
              << "      mark_from: "
              << connection.first.first
              << std::endl;
      outputFile
              << "      mark_to: "
              << connection.first.second
              << std::endl;
      outputFile
              << "      in_no: "
              << connection.second
              << std::endl;
    }

    outputFile << "- neuron_innovations: " << std::endl;
    for (const auto &neuron : neuronInnovations_)
    {
      outputFile << "  - neuron_innovation: " << std::endl;
      outputFile
              << "      conn_split: "
              << neuron.first.first
              << std::endl;
      outputFile << "      ntype: " << neuron.first.second << std::endl;
      outputFile << "      in_nos: " << std::endl;
      for (const auto in_no : neuron.second)
      {
        outputFile << "      - in_no: " << in_no << std::endl;
      }
    }
    outputFile.close();
  }

  /////////////////////////////////////////////////
  void Mutator::LoadInnovationsFromFirst(const std::string &_path)
  {
    YAML::Node yaml = YAML::LoadFile(_path);
    if (yaml.IsNull())
    {
      std::cerr << "Error: Load first innovations:" << _path << std::endl;
      return;
    }
    this->innovationNumber_ = yaml[0]["in_no"].as< size_t >();
    this->connectionInnovations_.clear();
    for (const auto &connection : yaml[1]["connection_innovations"])
    {
      auto innovation = connection["connection_innovation"];

      auto from = innovation["mark_from"].as< size_t >();
      auto to = innovation["mark_to"].as< size_t >();
      auto inNum = innovation["in_no"].as< size_t >();

      this->connectionInnovations_.insert({{from, to}, inNum});
    }
    this->neuronInnovations_.clear();
    for (const auto &neuron : yaml[2]["neuron_innovations"])
    {
      auto innovation = neuron["neuron_innovation"];
      auto split = innovation["conn_split"].as< size_t >();
      auto nType = static_cast<Neuron::Ntype>(innovation["ntype"].as< size_t >());
      std::vector< size_t > inNums;
      for (const auto &inNum : innovation["in_nos"])
      {
        inNums.push_back(inNum["in_no"].as< size_t >());
      }
      this->neuronInnovations_.insert({{split, nType}, inNums});
    }
  }

  void Mutator::LoadInnovationsFromSecond(const std::string &_path)
  {
    YAML::Node yaml = YAML::LoadFile(_path);
    if (yaml.IsNull())
    {
      std::cerr << "Error: Load second innovations:" << _path << std::endl;
      return;
    }

    // Map innvations from second to first
    this->secondToFirst_.clear();
    for (size_t innovations = 0; innovations < 10; ++innovations)
    {
      this->secondToFirst_[innovations] = innovations;
    }

    // Load innovations as for a first parent
    auto secondInnovationNumber_ = yaml[0]["in_no"].as< size_t >();

//    this->connectionInnovations_.clear();
    for (const auto &connection : yaml[1]["connection_innovations"])
    {
      auto innovation = connection["connection_innovation"];

      auto from = innovation["mark_from"].as< size_t >();
      auto to = innovation["mark_to"].as< size_t >();
      auto inNum = innovation["in_no"].as< size_t >();

      if (this->secondToFirst_.find(from) == this->secondToFirst_.end())
      {
        this->secondToFirst_[from] = from + this->innovationNumber_;
      }
      if (this->secondToFirst_.find(to) == this->secondToFirst_.end())
      {
        this->secondToFirst_[to] = to + this->innovationNumber_;
      }
      if (this->secondToFirst_.find(inNum) == this->secondToFirst_.end())
      {
        this->secondToFirst_[inNum] = inNum + this->innovationNumber_;
      }

      this->connectionInnovations_.insert(
              {{this->secondToFirst_[from], this->secondToFirst_[to]},
               this->secondToFirst_[inNum]});
    }
//    this->neuronInnovations_.clear();
    for (const auto &neuron : yaml[2]["neuron_innovations"])
    {
      auto innovation = neuron["neuron_innovation"];

      auto split = innovation["conn_split"].as< size_t >();
      auto nType = static_cast<Neuron::Ntype>(innovation["ntype"].as< size_t >());

      if (this->secondToFirst_.find(split) == this->secondToFirst_.end())
      {
        this->secondToFirst_[split] = split + this->innovationNumber_;
      }
      std::vector< size_t > inNums;
      for (const auto &inNum : innovation["in_nos"])
      {
        auto innov = inNum["in_no"].as< size_t >();
        if (this->secondToFirst_.find(innov) == this->secondToFirst_.end())
        {
          this->secondToFirst_[innov] = innov + this->innovationNumber_;
        }

        inNums.push_back(this->secondToFirst_[innov]);
      }
      this->neuronInnovations_.insert(
              {{this->secondToFirst_[split], nType},
               inNums});
    }

//    for (const auto &value : this->secondToFirst_)
//    {
//      std::cout << value.first << " <- " << value.second << std::endl;
//    }
//    std::cout << "innovationNumber_: " << innovationNumber_ << std::endl;

    this->innovationNumber_ += secondInnovationNumber_;
//    std::cout << "innovationNumber_: " << innovationNumber_ << std::endl;
//    std::exit(0);
  }

  void Mutator::MutateNeuronParams(
          GeneticEncodingPtr genotype,
          double probability,
          double sigma)
  {
    std::uniform_real_distribution< double > uniform(0, 1);
    if (not genotype->is_layered_)
    {
      for (const auto &neuron_gene : genotype->neurons_)
      {
        if (uniform(generator_) < probability)
        {
          auto neuron_params =
                  specification_[neuron_gene->neuron->type_].param_specs;
          if (not neuron_params.empty())
          {
            std::uniform_int_distribution< size_t > uniform_int(
                    0, neuron_params.size() - 1);
            auto param = neuron_params[uniform_int(generator_)];
            auto cur_val = neuron_gene->neuron->neuron_params[param.name];
            std::normal_distribution< double > normal(0, sigma);
            cur_val += normal(generator_);
            neuron_gene->neuron->SetParameters(cur_val, param);
          }
        }
      }
    }
    else
    {
      for (const auto &layer : genotype->layers_)
      {
        for (const auto &neuron_gene : layer)
        {
          if (uniform(generator_) < probability)
          {
            auto neuron_params =
                    specification_[neuron_gene->neuron->type_].param_specs;
            if (not neuron_params.empty())
            {
              std::uniform_int_distribution< size_t > uniform_int(
                      0, neuron_params.size() - 1);
              auto param = neuron_params[uniform_int(generator_)];
              auto cur_val = neuron_gene->neuron->neuron_params[param.name];
              std::normal_distribution< double > normal(0, sigma);
              cur_val += normal(generator_);
              neuron_gene->neuron->SetParameters(cur_val, param);
            }
          }
        }
      }
    }
  }

  /////////////////////////////////////////////////
  void Mutator::MutateWeights(
          GeneticEncodingPtr genotype,
          double probability,
          double sigma)
  {
    std::uniform_real_distribution< double > uniform(0, 1);
    std::normal_distribution< double > normal(0, sigma);
    for (const auto &connection_gene : genotype->connections_)
    {
      if (uniform(generator_) < probability)
      {
        connection_gene->ModifyWeight(normal(generator_));
      }
    }
  }

  /////////////////////////////////////////////////
  void Mutator::MutateStructure(
          GeneticEncodingPtr genotype,
          double probability)
  {
    std::uniform_real_distribution< double > uniform(0, 1);
    if (uniform(generator_) < probability)
    {
      if (genotype->connections_.empty())
      {
        MutateConnection(genotype, this->sigma_);
      }
      else
      {
        if (uniform(generator_) < 0.5)
        {
          MutateConnection(genotype, this->sigma_);
        }
        else
        {
          MutateNeuron(genotype, this->sigma_);
        }
      }
    }
  }

  /////////////////////////////////////////////////
  bool Mutator::MutateConnection(
          GeneticEncodingPtr genotype,
          double sigma)
  {
    if (not genotype->is_layered_)
    {
      std::uniform_int_distribution< size_t > choice(
              0, genotype->neurons_.size() - 1);
      auto from = genotype->neurons_[choice(generator_)];
      auto to = genotype->neurons_[choice(generator_)];
      auto markFrom = from->InnovationNumber();
      auto markTo = to->InnovationNumber();

      size_t num_attempts = 1;

      while (genotype->HasConnection(markFrom, markTo)
             or to->neuron->layer == Neuron::INPUT_LAYER)
      {
        from = genotype->neurons_[choice(generator_)];
        to = genotype->neurons_[choice(generator_)];
        markFrom = from->InnovationNumber();
        markTo = to->InnovationNumber();

        num_attempts++;
        if (num_attempts > maxAttempts_)
        {
          return false;
        }
      }
      std::normal_distribution< double > normal(0, sigma);
      this->AddConnection(
              markFrom,
              markTo,
              normal(generator_),
              genotype,
              "");
#ifdef CPPNEAT_DEBUG
      if (not genotype->is_valid()) {
          std::cerr << "add connection mutation caused invalid genotye" << std::endl;
          throw std::runtime_error("mutation error");
      }
#endif
      return true;
    }
    else
    {
      std::uniform_int_distribution< size_t > choice(
              0, genotype->num_neuron_genes() - 1);
      auto index_from =
              genotype->convert_index_to_layer_index(choice(generator_));
      auto index_to = genotype->convert_index_to_layer_index(choice(generator_));
      auto neuron_from = genotype->layers_[index_from.first][index_from.second];
      auto neuron_to = genotype->layers_[index_to.first][index_to.second];
      auto mark_from = neuron_from->InnovationNumber();
      auto mark_to = neuron_to->InnovationNumber();

      int num_attempts = 1;

      while (genotype->HasConnection(mark_from, mark_to)
             or index_from.first >= index_to.first)
      {
        index_from = genotype->convert_index_to_layer_index(choice(generator_));
        index_to = genotype->convert_index_to_layer_index(choice(generator_));
        neuron_from = genotype->layers_[index_from.first][index_from.second];
        neuron_to = genotype->layers_[index_to.first][index_to.second];
        mark_from = neuron_from->InnovationNumber();
        mark_to = neuron_to->InnovationNumber();

        num_attempts++;
        if (num_attempts > maxAttempts_)
        {
          return false;
        }
      }
      std::normal_distribution< double > normal(0, sigma);
      this->AddConnection(
              mark_from,
              mark_to,
              normal(generator_),
              genotype,
              "");
#ifdef CPPNEAT_DEBUG
      if (not genotype->is_valid()) {
          std::cerr << "add connection mutation caused invalid genotye" << std::endl;
          throw std::runtime_error("mutation error");
      }
#endif
      return true;
    }
  }

  /////////////////////////////////////////////////
  std::map< std::string, double > RandomParameters(
          Neuron::NeuronTypeSpec param_specs,
          double sigma)
  {
    std::map< std::string, double > params;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution< double > normal(0, sigma);
    for (Neuron::ParamSpec spec : param_specs.param_specs)
    {
      params[spec.name] = normal(generator);

    }
    return params;
  }

  /////////////////////////////////////////////////
  void Mutator::MutateNeuron(
          GeneticEncodingPtr genotype,
          double sigma)
  {
    assert(not genotype->connections_.empty());
    assert(not availableNeurons_.empty());
    if (not genotype->is_layered_)
    {
      std::uniform_int_distribution< size_t > choice1(
              0, genotype->connections_.size() - 1);
      auto split_id = choice1(generator_);
      auto split = genotype->connections_[split_id];

      auto old_weight = split->Weight();
      auto from = split->From();
      auto to = split->To();

      genotype->RemoveConnection(split_id);
      auto neuron_from = boost::dynamic_pointer_cast< NeuronGene >(
              genotype->Find(from))->neuron;
      auto neuron_to = boost::dynamic_pointer_cast< NeuronGene >(
              genotype->Find(to))->neuron;

      std::uniform_int_distribution< size_t > choice2(
              0, availableNeurons_.size() - 1);
      auto new_neuron_type = availableNeurons_[choice2(generator_)];

      auto new_neuron_params = RandomParameters(
              specification_[new_neuron_type],
              sigma);

      NeuronPtr neuron_middle(new Neuron(
              "augment" + std::to_string(innovationNumber_ + 1),
              Neuron::HIDDEN_LAYER,
              new_neuron_type,
              new_neuron_params));
      auto middle = AddNeuron(neuron_middle, genotype, split);
      this->AddConnection(
              from,
              middle,
              old_weight,
              genotype,
              "");
      this->AddConnection(
              middle,
              to,
              1.0,
              genotype,
              "");
    }
    else
    {
      std::uniform_int_distribution< size_t > choice1(
              0, genotype->connections_.size() - 1);
      auto split_id = choice1(generator_);
      auto split = genotype->connections_[split_id];

      auto old_weight = split->Weight();
      auto from = split->From();
      auto to = split->To();

      genotype->RemoveConnection(split_id);
      auto neuron_from = boost::dynamic_pointer_cast< NeuronGene >(
              genotype->Find(from))->neuron;
      auto neuron_to = boost::dynamic_pointer_cast< NeuronGene >(
              genotype->Find(to))->neuron;

      std::uniform_int_distribution< size_t > choice2(
              0, availableNeurons_.size() - 1);
      auto new_neuron_type = availableNeurons_[choice2(generator_)];

      auto new_neuron_params = RandomParameters(
              specification_[new_neuron_type],
              sigma);

      NeuronPtr neuron_middle(new Neuron(
              "augment" + std::to_string(innovationNumber_ + 1),
              Neuron::HIDDEN_LAYER,
              new_neuron_type,
              new_neuron_params));
      auto middle = AddNeuron(
              neuron_middle,
              genotype,
              split);
#ifdef CPPNEAT_DEBUG
      if (not genotype->is_valid()) {
          std::cerr << "add neuron mutation caused invalid genotye1" << std::endl;
          throw std::runtime_error("mutation error");
      }
#endif
      this->AddConnection(
              from,
              middle,
              old_weight,
              genotype,
              "");
#ifdef CPPNEAT_DEBUG
      if (not genotype->is_valid()) {
          std::cerr << "add neuron mutation caused invalid genotye2" << std::endl;
          throw std::runtime_error("mutation error");
      }
#endif
      this->AddConnection(
              middle,
              to,
              1.0,
              genotype,
              "");
    }
#ifdef CPPNEAT_DEBUG
    if (not genotype->is_valid()) {
        std::cerr << "add neuron mutation caused invalid genotye" << std::endl;
        throw std::runtime_error("mutation error");
    }
#endif
  }

  void Mutator::RemoveConnection(GeneticEncodingPtr genotype)
  {
    if (genotype->connections_.size() == 0)
    {
      return;
    }
    std::uniform_int_distribution< int > choice(
            0, genotype->connections_.size() - 1);
    genotype->connections_.erase(
            genotype->connections_.begin() + choice(generator_));
  }

  void Mutator::RemoveNeuron(GeneticEncodingPtr genotype)
  {
    std::vector< size_t > hidden_neuron_ids;
    for (size_t i = 0; i < genotype->neurons_.size(); i++)
    {
      if (genotype->neurons_[i]->neuron
                                    ->layer == Neuron::HIDDEN_LAYER)
      {
        hidden_neuron_ids.push_back(i);
      }
    }
    std::uniform_int_distribution< size_t > choice(
            0, hidden_neuron_ids.size() - 1);
    auto gene_id = hidden_neuron_ids[choice(generator_)];
    auto neuron_gene = genotype->neurons_[gene_id];
    auto neuron_mark = neuron_gene->InnovationNumber();

    std::vector< int > bad_connections;
    for (unsigned int i = 0; i < genotype->connections_.size(); i++)
    {
      if (genotype->connections_[i]->From() == neuron_mark
          or genotype->connections_[i]->To() == neuron_mark)
      {
        bad_connections.push_back(i);
      }
    }
    for (size_t i = bad_connections.size() - 1; i >= 0; i--)
    {
      genotype->RemoveConnection(bad_connections[i]);
    }
    genotype->RemoveNeuron(gene_id);
  }

  size_t Mutator::AddNeuron(
          const NeuronPtr neuron,
          const GeneticEncodingPtr genotype,
          const ConnectionGenePtr split)
  {
    auto splitConnectionId = split->InnovationNumber();
    std::pair< size_t , Neuron::Ntype > neuron_pair(
            splitConnectionId,
            neuron->type_);
    if (neuronInnovations_.find(neuron_pair) not_eq neuronInnovations_.end())
    {
      size_t i = 0;
      while (i < neuronInnovations_[neuron_pair].size()
             and genotype->Find(neuronInnovations_[neuron_pair][i])
                 not_eq nullptr)
      {
        i++;
      }
      //some previous innovation is not are already present in the genome-> add a it
      if (i < neuronInnovations_[neuron_pair].size())
      {
        NeuronGenePtr new_neuron_gene(new NeuronGene(
                neuron,
                neuronInnovations_[neuron_pair][i],
                true,
                "none",
                -1));
        if (not genotype->is_layered_)
        {
          genotype->AddNeuron(new_neuron_gene);
        }
        else
        {
          auto mark_from = split->From();
          auto mark_to = split->To();
          auto index_from = genotype->convert_in_to_layer_index(mark_from);
          auto index_to = genotype->convert_in_to_layer_index(mark_to);
          assert(index_from.first < index_to.first);
          genotype->AddNeuron(
                  new_neuron_gene,
                  index_from.first + 1,
                  //we need a new layer iff the two layers are "right next to each other"
                  index_from.first + 1 == index_to.first);

        }
        return new_neuron_gene->InnovationNumber();
      }
    }
    //new innovation -> add new neuron with new innovation number
    NeuronGenePtr new_neuron_gene(new NeuronGene(
            neuron,
            ++innovationNumber_,
            true,
            "none",
            -1));
    //in base case a new vector is constructed here
    neuronInnovations_[neuron_pair].push_back(innovationNumber_);
    if (not genotype->is_layered_)
    {
      genotype->AddNeuron(new_neuron_gene);
    }
    else
    {
      auto mark_from = split->From();
      auto mark_to = split->To();
      auto index_from = genotype->convert_in_to_layer_index(mark_from);
      auto index_to = genotype->convert_in_to_layer_index(mark_to);
      assert(index_from.first < index_to.first);
      genotype->AddNeuron(
              new_neuron_gene,
              index_from.first + 1,
              //we need a new layer iff the two layers are "right next to each other"
              index_from.first + 1 == index_to.first);
    }
    return new_neuron_gene->InnovationNumber();
  }

  size_t Mutator::AddConnection(
          const size_t _from,
          const size_t _to,
          const double _weight,
          const GeneticEncodingPtr _genotype,
          const std::string &_socket)
  {
    std::pair< size_t, size_t > innovation_pair(
            _from,
            _to);
    if (connectionInnovations_.find(innovation_pair) != connectionInnovations_.end())
    {
      auto found =
              _genotype->Find(connectionInnovations_[innovation_pair]);
      if (found != nullptr)
      {
        boost::dynamic_pointer_cast< ConnectionGene >(found)->setEnabled(true);
        return connectionInnovations_[innovation_pair];
      }
      else
      {
        ConnectionGenePtr new_conn_gene(new ConnectionGene(
                _to,
                _from,
                _weight,
                connectionInnovations_[innovation_pair],
                true,
                "none",
                -1,
                _socket));
        _genotype->AddConnection(new_conn_gene);
        return new_conn_gene->InnovationNumber();
      }
    }
    ConnectionGenePtr new_conn_gene(new ConnectionGene(
            _to,
            _from,
            _weight,
            ++innovationNumber_,
            true,
            "none",
            -1,
            _socket));
    connectionInnovations_[innovation_pair] = innovationNumber_;
    _genotype->AddConnection(new_conn_gene);
    return new_conn_gene->InnovationNumber();
  }
}