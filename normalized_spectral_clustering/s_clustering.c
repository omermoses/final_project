#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "stdlib.h"
#include "stdio.h"

typedef struct Cluster {
    int name, size;
    double *centroid;
    double *sum_of_obs;
} Cluster;

typedef struct Observation {
    double *values;
    Cluster *cluster;
} Observation;

static PyObject* run (PyObject *self, PyObject *args);
static void print_index(PyObject *index, int K);

static int spectral_clustering(PyObject *observations, int k, int n, int d, int max_iter);

static int spectral_clustering(PyObject *observations, int k, int n, int d, int max_iter) {

    Observation **input_values;
    int is_changed_from_last_run, found_k_clusters, number_of_iter,obs_num;
    Cluster **clusters_array;



    return 0;
}

static int** matrix_multiply(int *first[], int *second[], int n) {
    /*
    * Function to multiply two matrices
    *
    */

    int *p;
    int **result;

    p = calloc(n*n, sizeof(int));
    result = calloc(n,sizeof(int *));
    for( i=0 ; i<n ; i++ )
    result[i] = p+i*n;

   // Multiplying first and second matrices and storing it in result
   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
         for (int k = 0; k < n; ++k) {
            result[i][j] += first[i][k] * second[k][j];
         }
      }
   }
}


}


