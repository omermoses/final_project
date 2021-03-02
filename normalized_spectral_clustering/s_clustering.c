#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "stdlib.h"
#include "stdio.h"
#include <math.h>


static PyObject* run (PyObject *self, PyObject *args);
static void print_index(PyObject *index, int K);
static double** init_doubles_matrix(double n);
static double** create_diagonal_degree_matrix(double **wighted_matrix);
static double** calculate_L_norm(double **wighted_matrix, double **D, int n);
static double** diagonal_matrix_power(double **matrix, double power, int n);
static double** matrix_subtraction(double **first, double **second, int n);
static double** matrix_multiply(double **first, double **second, int n);
static double l2_norm(double *vector, int d);


static int spectral_clustering(PyObject *observations, int k, int n, int d, int max_iter);

static int spectral_clustering(PyObject *observations, int k, int n, int d, int max_iter) {

    Observation **input_values;
    int is_changed_from_last_run, found_k_clusters, number_of_iter,obs_num;
    Cluster **clusters_array;



    return 0;
}

static double** create_weighted_adjacency_matrix(double **data, int n){
    /*
    * create W
    * params: data- ndarray with the samples in the rows
    *        n- number of samples
    * return: W- weighted_adjacency_matrix
    */

    double **result;

    result = init_doubles_matrix(n);

    for (int j=0; j<n; j++){

        col =
        col = np.linalg.norm(col, 2, axis=1)
        col = np.exp(-col / 2)
        W[:, j] = col
    np.fill_diagonal(W, val=0.0)  # zero out the diagonal
    return W
    }

}

static double** create_diagonal_degree_matrix(double **wighted_matrix){
    /*
    create D
    params: wighted matrix
            n- number of samples
        return: D - diagonal degree matrix

    */

    int row, col, i;
    double sum=0.0;
    double **result;

    result = init_doubles_matrix(n);

    for (row=0; row<n; row++){
        for (col=0; col<n; col++){
            sum += w[row][col];

        }
        result[row][row] = sum;
        sum = 0;
    }

    return result;
}

static double** calculate_L_norm(double **wighted_matrix, double **D, int n) {
    /*
    * create L normalized
    * params: wighted matrix
    *         n - number of samples
    * return: result - The normalized L
    */

    int row, col, i;
    double L;
    double **temp;
    double **D_times_half;
    double **result;

    result = init_doubles_matrix(n);
    L = matrix_subtraction(D - wighted_matrix);


    D_times_half = diagonal_matrix_power(D, -0.5, n);
    temp = matrix_multiply(D_times_half, L);
    result = matrix_multiply(temp, D_times_half);

    return result;
}

static double** matrix_subtraction(double **first, double **second, int n) {

    int row, col;
    double **result;

    result = init_doubles_matrix(n);

    for (row=0; row<n; row++){
        for(col=0; col<n; col++){
            result[row][col] = first[row][col] - second[row][col];
        }
    }

    return result;

}

static double** matrix_multiply(double **first, double **second, int n) {
    /*
    * Function to multiply two matrices
    *
    */

    int row, col, i, k;
    double **result;

    result = init_doubles_matrix(n);

    for (row = 0; row < n; row++) {
        for (col = 0; col < n; col++) {
            for ( k = 0; k < n; k++) {
                result[row][col] += first[row][k] * second[k][col];
            }
        }
    }

    return result;
}

static double** diagonal_matrix_power(double **matrix, double power, int n) {

    int row, col;

    for (row=0; row<n; row ++){
        matrix[row][row] = pow(matrix[row][row], power);
    }

}

static double** init_doubles_matrix(int n) {

    double **matrix;

    matrix = calloc(n, sizeof(double *));
    if (matrix == NULL) {
        fprintf(stderr, "Failed while allocating memory");
        exit(1);
    }
    for( i=0 ; i < n ; i++ ) {
        matrix[i] = calloc(n, sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Failed while allocating memory");
            exit(1);
        }
    }

    return matrix;
}

static void gram_schmidt(double** matrix_a, double** matrix_q, double** matrix_r, int n){
    /*
        calculate Modified Gram Schmidt
        params: A- ndarray of size nXn with dtype='float64'
                n- A dim
        return: Q- orthogonal matrix
                R- diagonal matrix
    */
    for (i=0; i<n; i++){
        get
        matrix_r[i][i]=l2_norm()
    }
}

static double l2_norm(double *vector, int d){
    // calculate L2 norm
    int index;
    double dist;
    dist = 0;

    for (index =0; index < d; index++) {
        dist += pow(vector[index],2)
    }
    return pow(dist, 0.5);
}


