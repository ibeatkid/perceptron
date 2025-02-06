#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>              // Added for time()
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl_blas.h>
double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Derivative of the sigmoid function with respect to its input (x). 
   If you stored the "activated output", you can use s*(1-s). */
double sigmoid_derivative(const double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

typedef struct {
    int numLayers;   
    int *layerSizes;  
    gsl_matrix **weights;
    gsl_vector **biases;
} NeuralNetwork;

NeuralNetwork* createNetwork(int numLayers, int *layerSizes) {
    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        fprintf(stderr, "Failed to allocate memory for NeuralNetwork.\n");
        exit(EXIT_FAILURE);
    }
    
    net->numLayers = numLayers;
    net->layerSizes = (int*)malloc(numLayers * sizeof(int));
    if (!net->layerSizes) {
        fprintf(stderr, "Failed to allocate memory for layerSizes.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < numLayers; i++){
        net->layerSizes[i] = layerSizes[i];
    }
    
    net->weights = (gsl_matrix**)malloc(numLayers * sizeof(gsl_matrix*));
    net->biases  = (gsl_vector**)malloc(numLayers * sizeof(gsl_vector*));
    if (!net->weights || !net->biases) {
        fprintf(stderr, "Failed to allocate memory for weights or biases.\n");
        exit(EXIT_FAILURE);
    }
    
    /* Initialize GSL random number generator for weight initialization */
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, (unsigned long) time(NULL));
    
    /* Initialize weights and biases for layers 1 to numLayers-1 */
    for (int i = 1; i < numLayers; i++) {
        /* Create a weight matrix of size [ layerSizes[i], layerSizes[i-1] ] */
        net->weights[i] = gsl_matrix_alloc(net->layerSizes[i], net->layerSizes[i-1]);
        /* Create a bias vector of size [ layerSizes[i] ] */
        net->biases[i]  = gsl_vector_alloc(net->layerSizes[i]);
        
        /* Random initialization of weights and biases using a small random range */
        for (int r = 0; r < net->layerSizes[i]; r++) {
            for (int c = 0; c < net->layerSizes[i-1]; c++) {
                double w = gsl_ran_gaussian(rng, 0.1); // small random weight
                gsl_matrix_set(net->weights[i], r, c, w);
            }
            double b = gsl_ran_gaussian(rng, 0.1); // small random bias
            gsl_vector_set(net->biases[i], r, b);
        }
    }
    
    /* Set index 0 to NULL as it is not used */
    net->weights[0] = NULL;
    net->biases[0]  = NULL;
    
    gsl_rng_free(rng);
    return net;
}
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
gsl_vector** forwardPass(NeuralNetwork *net, const gsl_vector *inputs) {
    /* We will store the activations for each layer in an array of gsl_vector*. */
    gsl_vector **activations = (gsl_vector**)malloc(net->numLayers * sizeof(gsl_vector*));
    
    /* Copy the inputs into activations[0] as the input layer output. */
    activations[0] = gsl_vector_alloc(net->layerSizes[0]);
    gsl_vector_memcpy(activations[0], inputs);
    
    /* For each subsequent layer i, compute z = W * activation[i-1] + b, 
       then activation[i] = sigmoid(z). */
    for (int i = 1; i < net->numLayers; i++) {
        activations[i] = gsl_vector_alloc(net->layerSizes[i]);
        
        /* Instead of nested loops, first copy the bias vector into activations[i] */
        gsl_vector_memcpy(activations[i], net->biases[i]);
        
        /* Then perform the matrix-vector multiplication:
           activations[i] = net->weights[i] * activations[i-1] + activations[i] (which holds the biases) */
        gsl_blas_dgemv(CblasNoTrans,     // No transpose on weights
                       1.0,              // alpha multiplier
                       net->weights[i],  // weight matrix for layer i
                       activations[i-1], // activation vector from previous layer
                       1.0,              // beta multiplier (to add the bias already in activations[i])
                       activations[i]);  // result is stored back in activations[i]
        
        /* activation function */
        for (int r = 0; r < net->layerSizes[i]; r++) {
            double z = gsl_vector_get(activations[i], r);
            gsl_vector_set(activations[i], r, sigmoid(z));
        }
    }
    
    return activations;
}
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


#endif /* NEURALNETWORK_H */
