#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    double *arr;  // Array to store weights or features
} data;

/**
 * Computes the dot product of two vectors a.arr and b.arr,
 * each of length (n+1) (0..n).
 */
double dotproduct(const data a, const data b, const int n)
{
    double result = 0.0;
    // We loop from 0 up to n inclusive (n+1 elements total)
    for (int i = 0; i <= n; i++) {
        result += a.arr[i] * b.arr[i];
    }
    return result;
}

/**
 * Frees allocated memory for the dataset and the weights.
 * - numberofdatapoint: how many dataset rows were allocated
 * - numberofweights:   how many weight-arrays were allocated
 */
void cleanup(data *dataset, int numberofdatapoint,
             data *weights, int numberofweights)
{
    // 1) Free each row in the dataset (if allocated)
    if (dataset) {
        for (int i = 0; i < numberofdatapoint; i++) {
            free(dataset[i].arr);
        }
        free(dataset);
    }

    // 2) Free each weight array, then free the entire weights pointer
    if (weights) {
        for (int i = 0; i < numberofweights; i++) {
            free(weights[i].arr);
        }
        free(weights);
    }
}

int main(void)
{
    int classCount = -1;
    int dim = -1;
    int numberofdatapoint = -1;

    printf("Enter number of classifications, dimension, and number of data points\n");
    printf("(must have space between them):\n");

    if (scanf("%d %d %d", &classCount, &dim, &numberofdatapoint) != 3) {
        printf("Error reading input parameters\n");
        return 1;
    }

    // consume leftover newline
    getchar();

    // Validate input parameters
    if (classCount <= 0 || dim <= 0 || numberofdatapoint <= 0) {
        printf("Invalid input parameters. All values must be positive.\n");
        return 1;
    }

    // Allocate memory for weights (an array of data)
    data *weights = (data *)malloc(classCount * sizeof(data));
    if (!weights) {
        printf("Memory allocation failed for weights.\n");
        return 1;
    }

    // Initialize each weights[i].arr
    for (int i = 0; i < classCount; i++) {
        // Typically we include bias, so we want dim+1:
        weights[i].arr = (double *)calloc(dim + 1, sizeof(double));
        if (!weights[i].arr) {
            printf("Memory allocation failed for weights[%d].arr\n", i);
            // Clean up what was allocated up to index i
            cleanup(NULL, 0, weights, i);
            return 1;
        }
    }

    printf("Enter data points. Please end with a classification label (integer).\n");

    // Allocate memory for dataset (array of `data`)
    data *dataset = (data *)malloc(numberofdatapoint * sizeof(data));
    if (!dataset) {
        printf("Memory allocation failed for dataset.\n");
        // Clean up all weights
        cleanup(NULL, 0, weights, classCount);
        return 1;
    }

    // Allocate each row: (dim+2) because:
    //  - 1 for the bias,
    //  - dim features,
    //  - 1 for the classification label
    for (int i = 0; i < numberofdatapoint; i++) {
        dataset[i].arr = (double *)malloc((dim + 2) * sizeof(double));
        if (!dataset[i].arr) {
            printf("Memory allocation failed for dataset[%d].arr.\n", i);
            // Clean up allocated dataset up to i, plus weights
            cleanup(dataset, i, weights, classCount);
            return 1;
        }
    }

    // Read each data point
    for (int i = 0; i < numberofdatapoint; i++) {
        // First element is the bias input = 1.0
        dataset[i].arr[0] = 1.0;

        // We read the next (dim + 1) values:
        //  - dim features,
        //  - plus 1 classification label
        printf("Enter data point %d (features followed by class label): ", i + 1);

        for (int j = 1; j <= dim + 1; j++) {
            if (scanf("%lf", &dataset[i].arr[j]) != 1) {
                printf("Failed to read value %d of data point %d.\n", j, i + 1);
                cleanup(dataset, numberofdatapoint, weights, classCount);
                return 1;
            }
        }
        getchar(); // consume leftover newline

        // Validate the class label
        int class_label = (int)dataset[i].arr[dim + 1];
        if (class_label < 0 || class_label >= classCount) {
            printf("Invalid class label. Must be between 0 and %d\n", classCount - 1);
            cleanup(dataset, numberofdatapoint, weights, classCount);
            return 1;
        }
    }

    printf("Initializing linear machine...\n");
    const int MAX_ITERATIONS = 1000;
    const double LEARNING_RATE = 0.1;

    bool misclassified;
    int iteration = 0;
    
    // Keep going until no misclassifications or we hit MAX_ITERATIONS
    do {
        misclassified = false;
        iteration++;

        // Check each data point
        for (int i = 0; i < numberofdatapoint; i++) {
            // Find the class with maximum activation
            double max_activation = -1e9;
            int predicted_class = -1;
            
            for (int j = 0; j < classCount; j++) {
                double activation = dotproduct(weights[j], dataset[i], dim);
                if (activation > max_activation) {
                    max_activation = activation;
                    predicted_class = j;
                }
            }

            // Get actual class
            int actual_class = (int)dataset[i].arr[dim + 1];

            // If misclassified, update weights
            if (predicted_class != actual_class) {
                misclassified = true;
                
                // Increase weights for correct class
                for (int k = 0; k <= dim; k++) {
                    weights[actual_class].arr[k] += LEARNING_RATE * dataset[i].arr[k];
                }
                
                // Decrease weights for predicted (wrong) class
                for (int k = 0; k <= dim; k++) {
                    weights[predicted_class].arr[k] -= LEARNING_RATE * dataset[i].arr[k];
                }
            }
        }

        if (iteration % 100 == 0) {
            printf("Completed iteration %d\n", iteration);
        }

    } while (misclassified && iteration < MAX_ITERATIONS);

    // Report results
    if (!misclassified) {
        printf("Converged after %d iterations!\n", iteration);
    } else {
        printf("Maximum iterations (%d) reached without convergence.\n", MAX_ITERATIONS);
    }

    // Print final weights
    printf("\nFinal weights:\n");
    for (int i = 0; i < classCount; i++) {
        printf("Class %d: ", i);
        for (int j = 0; j <= dim; j++) {
            printf("%.4f ", weights[i].arr[j]);
        }
        printf("\n");
    }

    // Clean up
    cleanup(dataset, numberofdatapoint, weights, classCount);
    return 0;
}