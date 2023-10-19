#include "feedforward.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

nn_t * create_neural_network(int num_layers, int * layer_sizes)
{
	nn_t * nn = (nn_t *)malloc(sizeof(nn_t));
	nn->num_layers = num_layers;
	nn->layers = (layer_t *)malloc(num_layers * sizeof(layer_t));

	for(int i = 0; i < num_layers; i++)
	{
		nn->layers[i].num_inputs = i == 0 ? layer_sizes[i] : layer_sizes[i - 1];
		nn->layers[i].num_neurons = layer_sizes[i];
		nn->layers[i].neurons = (neuron_t *)malloc(layer_sizes[i] * sizeof(neuron_t));

		for(int j = 0; j < layer_sizes[i]; j++)
		{
			nn->layers[i].neurons[j].weights = (double*)malloc(nn->layers[i].num_inputs * sizeof(double));

			/* Initialize weights! */
			for(int k = 0; k < nn->layers[i].num_inputs; k++)
			{
				nn->layers[i].neurons[j].weights[k] = 0.0;
			}

			/* Initialize bias. */
			nn->layers[i].neurons[j].bias = 0.0;
		}
	}

	return nn;
}

void feedforward(nn_t * nn, double * input, double * output)
{
    for(int i = 0; i < nn->num_layers; i++)
	{
        for(int j = 0; j < nn->layers[i].num_neurons; j++)
		{
            double sum = 0.0;
            for(int k = 0; k < nn->layers[i].num_inputs; k++)
			{
                sum += nn->layers[i].neurons[j].weights[k] * input[k];
            }
            sum += nn->layers[i].neurons[j].bias;
            output[j] = 1.0 / (1.0 + exp(-sum));
        }
        input = output;
    }
}

void destroy_neural_network(nn_t * nn)
{
	for(int i = 0; i < nn->num_layers; i++)
	{
		for(int j = 0; j < nn->layers[i].num_neurons; j++)
		{
			free(nn->layers[i].neurons[j].weights);
		}
		free(nn->layers[i].neurons);
	}
	free(nn->layers);
	free(nn);
}

void train_neural_network(nn_t * nn, double * input, double * target, double learning_rate)
{
	double * output = (double *)malloc(nn->layers[nn->num_layers - 1].num_neurons * sizeof(double));
	feedforward(nn, input, output);

	for(int i = nn->num_layers - 1; i >= 0; i--)
	{
		for(int j = 0; j < nn->layers[i].num_neurons; j++)
		{
			double error;
			if(i == nn->num_layers - 1)
			{
				error = target[j] - output[j];
			}
			else
			{
				error = 0.0;
				for(int k = 0; k < nn->layers[i + 1].num_neurons; k++)
				{
					error += nn->layers[i + 1].neurons[k].weights[j] * nn->layers[i + 1].neurons[k].bias;
				}
			}
			
			double delta = error * output[j] * (1.0 - output[j]);

			/* Update bias. */
			nn->layers[i].neurons[j].bias += learning_rate * delta;

			/* Update weights */
			for(int k = 0; k < nn->layers[i].num_inputs; k++)
			{
				nn->layers[i].neurons[j].weights[k] += learning_rate * delta * input[k];
			}
		}

		input = output;
		if(i > 0)
		{
			free(output);
			output = (double *)malloc(nn->layers[i - 1].num_neurons * sizeof(double));
			feedforward(nn, nn->layers[i - 1].neurons[0].weights, output);
		}
	}

	free(output);
}