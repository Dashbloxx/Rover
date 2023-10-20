#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#include "feedforward.h"

int main()
{
	int num_layers = 3;
	int layer_sizes[] = {2, 2, 1};

	/* Create a neural network object! */
	nn_t * nn;

	if(access("model.bin", F_OK) != -1)
	{
        nn = load_neural_network("model.bin");
    }
	else
	{
		nn = create_neural_network(num_layers, layer_sizes);
	}

	/* Obtain two inputs from stdin. */
	double input[2];
	printf("Enter two input values: ");
	scanf("%lf %lf", &input[0], &input[1]);

	/* Obtain what is expected from the neural network. */
	double target[1];
	printf("Enter the target output value: ");
	scanf("%lf", &target[0]);

	/* The higher the learning rate, the more accurate it is. */
	double learning_rate = 0.1;
	for(int epoch = 0; epoch < 1000; epoch++)
	{
		train_neural_network(nn, input, target, learning_rate);
	}

	/* Compute the neural network, using the feedforward function. */
	double output[1];
	feedforward(nn, input, output);

	/* Print the input & output values. */
	printf("Input: %lf, %lf\n", input[0], input[1]);
	printf("Output: %lf\n", output[0]);

	save_neural_network(nn, "model.bin");

	/* De-allocate the neural network object. */
	destroy_neural_network(nn);

	return 0;
}