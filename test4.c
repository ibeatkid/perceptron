#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

/* --------------------- Sigmoid Activation Function --------------------- */
double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Derivative of the sigmoid function with respect to its input (x). 
   If you stored the "activated output", you can use s*(1-s). */
double sigmoid_derivative(const double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

/* --------------------- Neural Network Data Structure --------------------- */
typedef struct {
    int numLayers;      /* total layers (including input and output) */
    int *layerSizes;    /* array storing the size for each layer, e.g. [input, hidden1, ..., output] */

    /* weights[i] is a gsl_matrix of size (layerSizes[i], layerSizes[i-1])
       representing the weight matrix from layer (i-1) to layer (i).
       biases[i] is a gsl_vector of size layerSizes[i]. */
    gsl_matrix **weights;
    gsl_vector **biases;
} NeuralNetwork;

/* --------------------- Helper Function to Create a Neural Network --------------------- */
NeuralNetwork* createNetwork(int numLayers, int *layerSizes) {
    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->numLayers = numLayers;
    net->layerSizes = (int*)malloc(numLayers * sizeof(int));
    
    for(int i = 0; i < numLayers; i++){
        net->layerSizes[i] = layerSizes[i];
    }
    
    net->weights = (gsl_matrix**)malloc(numLayers * sizeof(gsl_matrix*));
    net->biases  = (gsl_vector**)malloc(numLayers * sizeof(gsl_vector*));
    
    /* Initialize GSL random number generator for weight initialization */
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, (unsigned long) time(NULL));
    
    /* We only have weights/biases from layer 0->1, 1->2, ... up to numLayers-2 -> numLayers-1
       So the indexing goes from 1 to numLayers-1 */
    for(int i = 1; i < numLayers; i++) {
        /* Create a weight matrix of size [ layerSizes[i], layerSizes[i-1] ] */
        net->weights[i] = gsl_matrix_alloc(net->layerSizes[i], net->layerSizes[i-1]);
        /* Create a bias vector of size [ layerSizes[i] ] */
        net->biases[i]  = gsl_vector_alloc(net->layerSizes[i]);
        
        /* Random initialization of weights and biases. 
           Using a small random range can help with initial training stability. */
        for(int r = 0; r < net->layerSizes[i]; r++) {
            for(int c = 0; c < net->layerSizes[i-1]; c++) {
                double w = gsl_ran_gaussian(rng, 0.1); // small random
                gsl_matrix_set(net->weights[i], r, c, w);
            }
            double b = gsl_ran_gaussian(rng, 0.1);
            gsl_vector_set(net->biases[i], r, b);
        }
    }
    
    /* We do not use indices for layer 0 in weights/biases, 
       but let's set them to NULL for clarity */
    net->weights[0] = NULL;
    net->biases[0]  = NULL;
    
    gsl_rng_free(rng);
    return net;
}

/* --------------------- Free the Neural Network --------------------- */
void freeNetwork(NeuralNetwork *net) {
    if(!net) return;
    for(int i = 1; i < net->numLayers; i++){
        gsl_matrix_free(net->weights[i]);
        gsl_vector_free(net->biases[i]);
    }
    free(net->weights);
    free(net->biases);
    free(net->layerSizes);
    free(net);
}

/* --------------------- Forward Pass ---------------------
   - inputs: a gsl_vector of size layerSizes[0] (the input layer).
   - returns: an array of gsl_vector*, each with the neuron outputs for that layer.
     For convenience, out[0] = input vector, out[1] = activation of layer 1, etc.
   - The user must free these allocated vectors after use.
*/
gsl_vector** forwardPass(NeuralNetwork *net, const gsl_vector *inputs) {
    /* We will store the activations for each layer in an array of gsl_vector*. */
    gsl_vector **activations = (gsl_vector**)malloc(net->numLayers * sizeof(gsl_vector*));
    
    /* Copy the inputs into activations[0] as the input layer output. */
    activations[0] = gsl_vector_alloc(net->layerSizes[0]);
    gsl_vector_memcpy(activations[0], inputs);
    
    /* For each subsequent layer i, compute z = W * activation[i-1] + b, 
       then activation[i] = sigmoid(z). */
    for(int i = 1; i < net->numLayers; i++){
        activations[i] = gsl_vector_alloc(net->layerSizes[i]);
        /* Weighted sum */
        for(int r = 0; r < net->layerSizes[i]; r++){
            double sum = gsl_vector_get(net->biases[i], r);
            for(int c = 0; c < net->layerSizes[i-1]; c++){
                double w = gsl_matrix_get(net->weights[i], r, c);
                double a = gsl_vector_get(activations[i-1], c);
                sum += w * a;
            }
            /* activation function */
            gsl_vector_set(activations[i], r, sigmoid(sum));
        }
    }
    
    return activations;
}

/* --------------------- Backward Pass (Compute Gradients & Update) ---------------------
   - net: the neural network
   - activations: array of layer activations from forwardPass
   - target: a gsl_vector of size layerSizes[numLayers-1], the desired output
   - learningRate: step size for gradient descent
*/
void backwardPass(NeuralNetwork *net, gsl_vector **activations, const gsl_vector *target, double learningRate) {
    int L = net->numLayers;
    
    /* 1. Compute delta for output layer:
          delta_L = (a_L - y) * sigmoid'(z_L)
       But we didn’t store z_L. We can reconstruct z_L by taking the inverse of sigmoid 
       on a_L or we can compute the derivative via a_L*(1 - a_L). 
    */
    gsl_vector *delta[L]; // We'll store delta for each layer
    
    /* delta[L-1] has size layerSizes[L-1] */
    delta[L-1] = gsl_vector_alloc(net->layerSizes[L-1]);
    for(int i = 0; i < net->layerSizes[L-1]; i++){
        double aL = gsl_vector_get(activations[L-1], i);
        double y  = gsl_vector_get(target, i);
        double d = (aL - y) * (aL * (1.0 - aL));  // derivative of sigmoid
        gsl_vector_set(delta[L-1], i, d);
    }
    
    /* 2. Backpropagate deltas:
          delta[l] = ( (W[l+1])^T * delta[l+1] ) elementwise* sigmoid'(z_l)
       We'll similarly do the derivative via a_l*(1-a_l).
    */
    for(int l = L - 2; l >= 1; l--){
        delta[l] = gsl_vector_alloc(net->layerSizes[l]);
        for(int i = 0; i < net->layerSizes[l]; i++){
            double sum = 0.0;
            for(int j = 0; j < net->layerSizes[l+1]; j++){
                double w = gsl_matrix_get(net->weights[l+1], j, i);
                double d_next = gsl_vector_get(delta[l+1], j);
                sum += w * d_next;
            }
            double a_l = gsl_vector_get(activations[l], i);
            double d = sum * (a_l * (1.0 - a_l));
            gsl_vector_set(delta[l], i, d);
        }
    }
    
    /* 3. Update weights and biases using delta:
        For each layer l in 1..L-1:
           W[l] <- W[l] - eta * (delta[l] * a[l-1]^T)
           b[l] <- b[l] - eta * delta[l]
    */
    for(int l = 1; l < L; l++){
        /* Update each weight w_{l, r, c} where r indexes row (neuron in layer l),
           c indexes column (neuron in layer l-1). */
        for(int r = 0; r < net->layerSizes[l]; r++){
            for(int c = 0; c < net->layerSizes[l-1]; c++){
                double old_w = gsl_matrix_get(net->weights[l], r, c);
                double delta_r = gsl_vector_get(delta[l], r);
                double a_prev_c = gsl_vector_get(activations[l-1], c);
                
                double new_w = old_w - learningRate * delta_r * a_prev_c;
                gsl_matrix_set(net->weights[l], r, c, new_w);
            }
            /* Update bias */
            double old_b = gsl_vector_get(net->biases[l], r);
            double delta_r = gsl_vector_get(delta[l], r);
            double new_b = old_b - learningRate * delta_r;
            gsl_vector_set(net->biases[l], r, new_b);
        }
    }
    
    /* 4. Free delta vectors */
    for(int l = 1; l < L; l++){
        gsl_vector_free(delta[l]);
    }
}

/* --------------------- Main --------------------- */
int main(void){
    /* 1. Ask user for input dimension, output dimension, number of hidden layers, etc. */
    int inputSize, outputSize, hiddenLayers;
    printf("Enter the dimension of input data (number of features): ");
    scanf("%d", &inputSize);
    
    printf("Enter the dimension of output data (number of outputs): ");
    scanf("%d", &outputSize);
    
    printf("Enter the number of hidden layers: ");
    scanf("%d", &hiddenLayers);
    
    /* For a standard feed-forward net, we have:
       total layers = hiddenLayers + 2 (input and output). */
    int totalLayers = hiddenLayers + 2;
    int *layerSizes = (int*)malloc(sizeof(int) * totalLayers);
    
    /* layer 0 = input size */
    layerSizes[0] = inputSize;
    
    /* Ask user for number of neurons in each hidden layer */
    for(int i = 0; i < hiddenLayers; i++){
        int hSize;
        printf("Enter number of neurons in hidden layer %d: ", i+1);
        scanf("%d", &hSize);
        layerSizes[i+1] = hSize;
    }
    
    /* layer last = output size */
    layerSizes[hiddenLayers + 1] = outputSize;
    
    /* 2. Create the network */
    NeuralNetwork *net = createNetwork(totalLayers, layerSizes);
    
    /* 3. For demonstration, let's ask user for a single training example (x->y) 
          to do a single (or a few) steps of backprop. 
       In reality, you might load a dataset from file.
    */
    gsl_vector *x = gsl_vector_alloc(inputSize);
    gsl_vector *y = gsl_vector_alloc(outputSize);
    
    printf("\nEnter the input sample (space-separated, length = %d):\n", inputSize);
    for(int i = 0; i < inputSize; i++){
        double val;
        scanf("%lf", &val);
        gsl_vector_set(x, i, val);
    }
    
    printf("\nEnter the target output (space-separated, length = %d):\n", outputSize);
    for(int i = 0; i < outputSize; i++){
        double val;
        scanf("%lf", &val);
        gsl_vector_set(y, i, val);
    }
    
    /* 4. Training parameters: e.g., a simple single-epoch or multiple steps. */
    double learningRate = 0.1; 
    int epochs = 1000;
    
    for(int e = 0; e < epochs; e++){
        /* Forward pass */
        gsl_vector **activations = forwardPass(net, x);
        
        /* Compute MSE or any cost function at output (just for demonstration) */
        double mse = 0.0;
        for(int i = 0; i < outputSize; i++){
            double diff = gsl_vector_get(activations[net->numLayers - 1], i) - gsl_vector_get(y, i);
            mse += diff * diff;
        }
        mse /= outputSize;
        
        /* Backpropagation */
        backwardPass(net, activations, y, learningRate);
        
        /* Free the activation arrays from the forward pass */
        for(int i = 0; i < net->numLayers; i++){
            gsl_vector_free(activations[i]);
        }
        free(activations);
        
        /* (Optional) Print progress every 100 epochs */
        if((e+1) % 100 == 0) {
            printf("Epoch %d, MSE = %f\n", e+1, mse);
        }
    }
    
    /* 5. Test the trained network (forward pass again with the same x) */
    gsl_vector **finalActivations = forwardPass(net, x);
    gsl_vector *output = finalActivations[net->numLayers - 1];
    
    printf("\nTrained output after %d epochs:\n", epochs);
    for(int i = 0; i < outputSize; i++){
        printf("Output[%d] = %f\n", i, gsl_vector_get(output, i));
    }
    
    /* Clean up */
    for(int i = 0; i < net->numLayers; i++){
        gsl_vector_free(finalActivations[i]);
    }
    free(finalActivations);
    gsl_vector_free(x);
    gsl_vector_free(y);
    freeNetwork(net);
    free(layerSizes);
    
    return 0;

