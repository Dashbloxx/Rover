#pragma once

typedef struct
{
	double *weights;
	double bias;
} neuron_t;

typedef struct
{
	int num_inputs;
	int num_neurons;
	neuron_t * neurons;
} layer_t;

typedef struct
{
	int num_layers;
	layer_t * layers;
} nn_t;

nn_t * create_neural_network(int num_layers, int * layer_sizes);
void feedforward(nn_t * nn, double * input, double * output);
void destroy_neural_network(nn_t * nn);
void train_neural_network(nn_t * nn, double * input, double * target, double learning_rate);
void save_neural_network(nn_t * nn, const char * filename);
nn_t * load_neural_network(const char * filename);