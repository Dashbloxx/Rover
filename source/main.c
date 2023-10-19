#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define INPUT_SIZE 27  // 26 letters + 1 for space
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 27

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Define the neural network architecture
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];

    double input_weights[INPUT_SIZE][HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double hidden_bias[HIDDEN_SIZE];
    double output_bias[OUTPUT_SIZE];

    // Initialize weights and biases (randomly for simplicity)
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            input_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    char seed_text[100] = "hello";  // Seed text to start the generation

    // Text generation loop
    for (int i = 0; i < 200; i++) {
        // Encode the seed_text into the input
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = 0;
        }
        int seed_length = strlen(seed_text);
        for (int j = 0; j < seed_length; j++) {
            char c = seed_text[j];
            if (c >= 'a' && c <= 'z') {
                input[c - 'a'] = 1;
            } else if (c == ' ') {
                input[26] = 1;
            }
        }

        // Forward pass through the network
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden[j] = 0;
            for (int k = 0; k < INPUT_SIZE; k++) {
                hidden[j] += input[k] * input_weights[k][j];
            }
            hidden[j] += hidden_bias[j];
            hidden[j] = sigmoid(hidden[j]);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output[j] = 0;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                output[j] += hidden[k] * output_weights[k][j];
            }
            output[j] += output_bias[j];
            output[j] = sigmoid(output[j]);
        }

        // Sample the next character
        double rand_val = (double)rand() / RAND_MAX;
        double cumulative_prob = 0;
        char next_char = ' ';
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            cumulative_prob += output[j];
            if (rand_val < cumulative_prob) {
                if (j < 26) {
                    next_char = 'a' + j;
                } else {
                    next_char = ' ';
                }
                break;
            }
        }

        // Print the generated character
        printf("%c", next_char);

        // Update the seed_text for the next iteration
        memmove(seed_text, seed_text + 1, seed_length - 1);
        seed_text[seed_length - 1] = next_char;
    }

    printf("\n");

    return 0;
}
