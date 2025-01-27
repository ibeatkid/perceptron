#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    double *arr;  // Array to store weights
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

int main() {
    int n = -1;  // dimension (number of features)
    int k = -1;  // number of data points
    
    data weight1;

    // Get dimension
    printf("Enter dimension: ");
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Failed to read a valid positive dimension.\n");
        return 1;
    }
    getchar();  // consume leftover newline

    /**
     * We store n features + 1 bias = (n+1) weights total.
     * No need for (n+2) here, because we do NOT store the label in weight1.
     */
    weight1.arr = (double *)malloc((n + 1) * sizeof(double));
    if (weight1.arr == NULL) {
        printf("Memory allocation failed for weights.\n");
        return 1;
    }

    // Initialize weights to 0
    for (int i = 0; i <= n; i++) {
        weight1.arr[i] = 0.0;
    }

    // Get number of data points
    printf("Enter number of data points: ");
    if (scanf("%d", &k) != 1 || k <= 0) {
        printf("Failed to read valid number of data points.\n");
        free(weight1.arr);
        return 1;
    }
    getchar();  // consume leftover newline

    printf("Enter linearly separable data.\n");

    /**
     * Each data point has:
     *   index 0    -> the bias (always 1.0)
     *   indices 1..n -> the n feature values
     *   index (n+1) -> the label/classification (1 or -1)
     *
     * Hence each data point needs n+2 doubles.
     */
    double **dataset = (double **)malloc(k * sizeof(double *));
    if (dataset == NULL) {
        printf("Memory allocation failed for dataset.\n");
        free(weight1.arr);
        return 1;
    }

    // Allocate and read each data point
    for (int i = 0; i < k; i++) {
        dataset[i] = (double *)malloc((n + 2) * sizeof(double));
        if (dataset[i] == NULL) {
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(dataset[j]);
            }
            free(dataset);
            free(weight1.arr);
            printf("Memory allocation failed for data point %d.\n", i);
            return 1;
        }
    }

    // Read each data point
    for (int i = 0; i < k; i++) {
        printf("Enter data point %d (%d features + label):\n", i+1, n);
        
        // First element is the bias input = 1.0
        dataset[i][0] = 1.0;

        // Read n features and then the class label
        // j = 1..n are features, j = n+1 is the label
        for (int j = 1; j <= n + 1; j++) {
            if (scanf("%lf", &dataset[i][j]) != 1) {
                printf("Failed to read feature %d of data point %d.\n", j, i+1);
                // Clean up and exit
                for (int x = 0; x < k; x++) {
                    free(dataset[x]);
                }
                free(dataset);
                free(weight1.arr);
                return 1;
            }
        }
        getchar(); // consume leftover newline
    }

    printf("Starting Perceptron Learning Algorithm...\n");

    // Perceptron hyperparameters
    const int MAX_ITERATIONS = 1000;
    const double LEARNING_RATE = 0.1;

    bool misclassified;
    int iteration = 0;

    // Keep going until no misclassifications or we hit MAX_ITERATIONS
    do {
        misclassified = false;
        iteration++;

        // Check each data point
        for (int i = 0; i < k; i++) {
            // Temporarily wrap dataset[i] in our `data` struct
            data current_point;
            current_point.arr = dataset[i];

            // Compute dot product w^T x
            double prediction_value = dotproduct(weight1, current_point, n);

            // Predicted class sign
            int predicted_class = (prediction_value >= 0.0) ? 1 : -1;

            // Actual class is in dataset[i][n+1]
            int actual_class = (int)dataset[i][n + 1];

            // Check for misclassification and update
            if (predicted_class != actual_class) {
                misclassified = true;
                // Update each weight (including bias index 0)
                for (int j = 0; j <= n; j++) {
                    weight1.arr[j] += LEARNING_RATE * actual_class * dataset[i][j];
                }
            }
        }

        // Print out weights every 100 iterations to monitor progress
        if (iteration % 100 == 0) {
            printf("Iteration %d: Current Weights = ", iteration);
            for (int i = 0; i <= n; i++) {
                printf("%.2f ", weight1.arr[i]);
            }
            printf("\n");
        }

    } while (misclassified && iteration < MAX_ITERATIONS);

    // Final result
    if (iteration >= MAX_ITERATIONS) {
        printf("Algorithm did not converge after %d iterations.\n", MAX_ITERATIONS);
    } else {
        printf("Converged after %d iterations!\n", iteration);
        printf("Final weights:\n");
        for (int i = 0; i <= n; i++) {
            printf("w[%d] = %.2f  ", i, weight1.arr[i]);
        }
        printf("\n");
    }

    // Cleanup
    for (int i = 0; i < k; i++) {
        free(dataset[i]);
    }
    free(dataset);
    free(weight1.arr);

    return 0;
}