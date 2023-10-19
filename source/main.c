#include <stdio.h>

#include "feedforward.h"

int main()
{
	int layer_sizes[] = {2, 3, 1};
	int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
	nn_t * nn = create_neural_network(num_layers, layer_sizes);

	double input[] = {0.5, 0.7};
	double output[1];

	feedforward(nn, input, output);

	printf("Output: %lf\n", output[0]);

	destroy_neural_network(nn);

	return 0;
}
