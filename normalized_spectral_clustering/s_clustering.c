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

typedef struct {
  double **array;    /* Pointer to an array of type TYPE */
  int rows;       /* Number of rows */
  int cols;       /* Number of columns */
} matrix;

//void qr_iteration(matrix *A, int n){
//    int i;
//    matrix *Q_bar=create_matrix(n,n);
//    create_identity_matrix(Q);
//    for (i=0; i<n; i++){
//        gram_schmidt(A,n);
////        a=np.einsum('ij,jk', R, Q)
////        # a = R @ Q
////        q_temp=np.einsum('ij,jk', q, Q)
////        # q_temp = q @ Q
////        dist = np.absolute(q) - np.absolute(q_temp)
////        # if (-e <= dist).all() and (dist <= e).all():
////        if (np.absolute(dist) <= e).all():
////            break
////        q = q_temp
////    return a, q
//    }
//}


void gram_schmidt(matrix *A, int n){
  matrix *Q = create_matrix(n,n);
  matrix *R = create_matrix(n,n);
  QRdecompose(A,Q,R);
}

/* Decomposes the matrix A into QR */
void QRdecompose(matrix *A, matrix *Q, matrix *R) {

  /* Using the Gram-Schmidt process */

  /* Temporary vector T and S used in calculations */
  matrix *T = create_matrix(A->rows, 1);
  matrix *S = create_matrix(A->rows, 1);

  for (int i = 0; i < A->cols; i++) {
    //Qi = Ui
    matrix_copy_column(A,i,Q,i);
    //r[i,i] = ||Qi||
    R->array[i][i] = l2_norm(Q,i);
     //Qi = Qi/r[i,i]
    matrix_column_divide(Q,i,R->array[i][i]);
    for (int j = 0; j < i; j++) {
      //r[i,j] = Qi^T * Uj
      matrix_copy_column(Q,i,T,0);
      matrix_copy_column(A,j,S,0);
      TYPE r = 0;
      for (int k=0; k<A->rows; k++) {
        r += T->array[k][0] * S->array[k][0];
      }

      R->array[i][j] = r;
      matrix_column_subtract(Q,j,matrix_column_multiply(T,0,r),0);
    }
  }

  //free_matrix(T);
  //free_matrix(S);
}

/* Creates a matrix and returns a pointer to the struct */
matrix* create_matrix(int rows, int cols) {

  /* Allocate memory for the matrix struct */
  matrix *array = malloc(sizeof(matrix));

  array->rows = rows;
  array->cols = cols;

  /* Allocate enough memory for all the rows in the first matrix */
  array->array = calloc(rows, sizeof(double*));

  /* Enough memory for all the columns */
  for (int i=0; i<rows; i++) {
    array->array[i] = calloc(cols,sizeof(double));
  }

  return array;
}

/* Creates a matrix from a stack based array and returns a pointer to the struct */
matrix* create_matrix_from_array(int rows, int cols, double m[][cols]) {

  /* Allocate memory for the matrix struct */
  matrix *array = malloc(sizeof(matrix));
  array->rows = rows;
  array->cols = cols;

  /* Allocate memory for the matrix */
  array->array = malloc(sizeof(double*) * rows);

  /* Allocate memory for each array inside the matrix */
  for (int i=0; i<rows; i++) {
    array->array[i] = malloc(sizeof(double) * cols);
  }

  /* Populate the matrix with m's values */
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      array->array[row][col] = m[row][col];
    }
  }

  return array;
}
/* Copies a matrix column from source at column col1 to dest at column col2 */
void matrix_copy_column(matrix *m1, int col1, matrix *m2,int col2) {
  for (int i=0; i<m1->rows; i++) {
    m2->array[i][col2] = m1->array[i][col1];
  }
}

/* Subtracts m2's column c2 from m1's column c1 */
matrix* matrix_column_subtract(matrix *m1, int c1, matrix *m2, int c2) {
  for (int i=0; i<m1->rows; i++) {
      m1->array[i][c1] -= m2->array[i][c2];
  }
  return m1;
}

/* Returns the length of the vector column in m */
double l2_norm(matrix *m,int column) {
  double length = 0;
  for (int row=0; row<m->rows; row++) {
    length += m->array[row][column] * m->array[row][column];
  }
  return sqrt(length);
}

/* Divides the matrix column c in m by k */
matrix* matrix_column_divide(matrix *m, int c, double k) {
  for (int i=0; i<m->rows; i++) {
    m->array[i][c] /= k;
  }
  return m;
}

/* Multiplies the matrix column c in m by k */
matrix* matrix_column_multiply(matrix *m, int c, double k) {
  for (int i=0; i<m->rows; i++) {
    m->array[i][c] *= k;
  }
  return m;
}

void create_identity_matrix(matrix* m){
    for(i=0;i<m->rows;i++){
        for(j=0;j<m->columns;j++){
            if(i==j){
                m[i][j] = 1;
            }
        }
    }
}

///* Debugging purposes only */
//void print_matrix(matrix *m) {
//  for (int row = 0; row < m->rows; row++) {
//    printf("[");
//    for (int col = 0; col < m->cols - 1; col++) {
//      printf("%7.3f, ", m->array[row][col]);
//    }
//    printf("%7.3f", m->array[row][m->cols-1]);
//    printf("]\n");
//  }
//  printf("\n");
//}

//static double l2_norm(double *vector, int d){
//    // calculate L2 norm
//    int index;
//    double dist;
//    dist = 0;
//
//    for (index =0; index < d; index++) {
//        dist += pow(vector[index],2)
//    }
//    return pow(dist, 0.5);
//}


