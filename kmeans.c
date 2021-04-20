/*
kmeans.c executes the calculation for k-means++ algorithm of the project.
k-means++ algorithm calculates k clusters fron n samples.
The module triggered by kmeans.py connected via capi.c file.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.h" //include kmeans header
#include "stdlib.h"
#include "stdio.h"

#define EPS 0.0001

typedef struct Cluster {
    int name, size;
    double *centroid;
    double *sum_of_obs;
} Cluster;

typedef struct Observation {
    double *values;
    Cluster *cluster;
} Observation;

int kmeans(PyObject *observations, int k, int n, int d, int max_iter, long* index, long* obs_cluster_array);

static void convert_obs(Observation **input_values, PyObject *observations, int N, int d);

static void create_clusters_array(long *obs_cluster_array, Observation **input_values, long* index, int n);

static void clean(Observation **observations, int n, Cluster **cluster_array, int k);

static int init(Observation **observations, int n, int d);

static void find_closest_cluster(Observation *observation, Cluster **clusters_array, int k, int d);

static double squared_euclidean_distance(double *observation, double *centroid, int d);

static void observation_sum(double *sum_of_obs, double *observation_values, int d);

static void observation_sub(double *sum_of_obs, const double *observation_values, int d);

static void remove_obs(Observation *observation,int d) ;

static int update_centroid(Cluster **clusters_array, int k, int d);

static void insert_obs(Observation *observation, Cluster *best_cluster, int d);

static int create_k_clusters(Observation **observations, Cluster **clusters_array, int k, int d);

static void copy(const double *values, double *sum_of_obs, int d);

int kmeans(PyObject *observations, int k, int n, int d, int max_iter, long* index, long* obs_cluster_array) {
    /*
    returns an array of n ints where arr[i]=cluster of obs i in the original data
    */
    Observation **input_values;
    int is_changed_from_last_run, found_k_clusters, number_of_iter,obs_num;
    Cluster **clusters_array;

    input_values=malloc(n*sizeof(Observation));
    if (input_values==NULL){
        return -1;
        }

    if (init(input_values, n, d)==-1){
        return -1;}
    convert_obs(input_values, observations, n, d);

    is_changed_from_last_run= 1;
    found_k_clusters = 0;
    number_of_iter = 1;
    obs_num = k;

    clusters_array=malloc(k*sizeof(Cluster));
    if (clusters_array==NULL){
       return -1;
        }

    while (is_changed_from_last_run == 1 && (number_of_iter <= max_iter)) {
        /*main loop*/
        if (found_k_clusters == 1) {
            /*k cluster have been initiated*/
            find_closest_cluster(input_values[obs_num], clusters_array, k, d);
            obs_num+=1;

        } else {
            /*initiate k clusters*/
            if (create_k_clusters(input_values, clusters_array, k, d)==-1){
            return -1;
            }
            found_k_clusters = 1;
        }
        if (obs_num == n) {
            /*start new iteration*/
            is_changed_from_last_run=update_centroid(clusters_array,k,d);
            obs_num = 0;
            number_of_iter += 1;
        }
    }
    create_clusters_array(obs_cluster_array, input_values, index, n);
    clean(input_values, n, clusters_array, k);
    return 0;
}

static void convert_obs(Observation **input_values, PyObject *observations, int n, int d){
    /*
    * convert the PyObject type observations to arrays of type double
    */
    int i, j;
    PyObject *obs, *val;
    Py_ssize_t obs_num, obs_size;
    obs_num= PyList_Size(observations);
    for (i=0; i<obs_num; i++){
        obs=PyList_GetItem(observations, i);
        if (!PyList_Check(obs)){
           PyErr_Format(PyExc_ValueError,"%s,\nobservation should be a list",ERROR_MSG);
        }
        obs_size=PyList_Size(obs);
        for (j=0; j<obs_size; j++){
            val=PyList_GetItem(obs, j);
            if (!PyFloat_Check(val)){
                PyErr_Format(PyExc_ValueError, "%s,\nitems should be floats",ERROR_MSG);
                }
            input_values[i]->values[j]=PyFloat_AsDouble(val);
            if (input_values[i]->values[j]== -1 && PyErr_Occurred()){
            /* double too big to fit in a C float, bail out */
                PyErr_Format(PyExc_ValueError, "%s,PyFloat_AsDouble() failed\n",ERROR_MSG);
            }
        }
    }
}

static void create_clusters_array(long *obs_cluster_array, Observation **input_values, long* index, int n){
    // create an array where arr[original index of observation]=the cluster obs belongs to
    int i;
    for (i=0; i<n; i++){
        obs_cluster_array[index[i]]=input_values[i]->cluster->name;
    }
}

static int create_k_clusters(Observation **observations, Cluster **clusters_array, int k, int d) {
    /* create clusters array */
    int index;
    for (index = 0; index < k; index++) {
        clusters_array[index] = malloc(sizeof(Cluster));
        if (clusters_array[index]==NULL){ return -1; }
        clusters_array[index]->name = index;
        clusters_array[index]->size = 1;
        clusters_array[index]->centroid = calloc(d, sizeof(double ));
        if (clusters_array[index]->centroid==NULL){ return -1; }
        copy(observations[index]->values, clusters_array[index]->centroid, d);
        clusters_array[index]->sum_of_obs = calloc(d, sizeof(double ));
        if (clusters_array[index]->sum_of_obs ==NULL){
            return -1;
        }
        copy(observations[index]->values, clusters_array[index]->sum_of_obs, d);
        observations[index]->cluster=clusters_array[index];
    }
    return 0;
}

static void copy(const double *values, double *sum_of_obs, int d) {
    int i;
    for (i=0; i<d; i++){
        sum_of_obs[i]=values[i];
    }
}

static int init(Observation **observations, int n, int d) {
    /* allocate memory for observations */
    int i;
    for (i = 0; i < n; i++) {
        observations[i] = malloc(sizeof(Observation));
        if (observations[i]==NULL){
            return -1;
        }
        observations[i]->values = (double *) calloc(d, sizeof(double));
        if (observations[i]->values==NULL){
            return -1;
            }
        observations[i]->cluster = NULL;
    }
    return 0;
}

static void clean(Observation **observations, int n, Cluster **cluster_array, int k) {
    /* free all the memory */
    int i,j;
    for (i = 0; i < n; i++) {
        free(observations[i]->values);
        free(observations[i]);
    }
    free(observations);

    for (j = 0; j < k; j++) {
        free(cluster_array[j]->sum_of_obs);
        free(cluster_array[j]->centroid);
        free(cluster_array[j]);
    }
    free(cluster_array);
}


static void find_closest_cluster(Observation *observation, Cluster **clusters_array, int k, int d) {
    /*find closest cluster for observation (of class Observation)
    size of clusters_array is K, each index is of struct Cluster */

    int index;
    double min_dist;
    double dist;
    Cluster *best_cluster=NULL;

    min_dist = squared_euclidean_distance(observation->values, clusters_array[0]->centroid, d);
    best_cluster = clusters_array[0];

    for (index=1; index < k; index++) {
        dist = squared_euclidean_distance(observation->values, clusters_array[index]->centroid, d);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = clusters_array[index];
        }
    }
    if (observation->cluster != NULL){
        if (observation->cluster->name == best_cluster->name){ return; }
        remove_obs(observation, d);
    }
    insert_obs(observation, best_cluster, d);
}

static double squared_euclidean_distance(double *observation, double *centroid, int d){
    /*find cluster’s centroid using squared Euclidean distance
    observation and centroid are lists of size D*/
    int index;
    double dist;
    double temp;
    dist = 0;

    for (index =0; index < d; index++) {
        temp = (observation[index] - centroid[index]);
        dist += (temp*temp);
    }
    return dist;
}

static void observation_sum(double *sum_of_obs, double *observation_values, int d){
    /* sum_of_obs is a list in len D that sums all observations that belongs to the cluster*/
    int index;
    for (index=0; index<d; index++){
        sum_of_obs[index] += observation_values[index];
    }
}

static void insert_obs(Observation *observation, Cluster *best_cluster, int d) {
    observation_sum(best_cluster->sum_of_obs, observation->values, d);
    best_cluster->size++;
    observation->cluster = best_cluster;
}

static void observation_sub(double *sum_of_obs, const double *observation_values, int d) {
    /*update sum_of_obs sum_of_obs is a list in len D that sums all observations that belongs to the cluster*/
    int index;
    for (index=0; index < d; index++){
        *(sum_of_obs + index) -= *(observation_values + index);
    }
}

static void remove_obs(Observation *observation,int d) {
    observation->cluster->size -= 1;
    observation_sub(observation->cluster->sum_of_obs, observation->values, d);
}

static int update_centroid(Cluster **clusters_array, int k, int d){
    /*update centroid using the sum of observations that belongs to the cluster */
    int dpoint;
    int cluster_index;
    int is_changed;
    double temp_calc;
    is_changed = 0;

    for (cluster_index=0; cluster_index<k ;cluster_index++) {
        /*iterate over the clusters*/
        Cluster *current_cluster;
        current_cluster= clusters_array[cluster_index];
        for (dpoint=0; dpoint<d; dpoint++){
            temp_calc = current_cluster->sum_of_obs[dpoint]/(float)current_cluster->size;
            if (temp_calc - current_cluster->centroid[dpoint] < EPS || temp_calc - current_cluster->centroid[dpoint] > -EPS) {
                /*check if the centroid in place dpoint should be updated*/
                current_cluster->centroid[dpoint] = temp_calc;
                is_changed = 1;
            }
        }
    }
    return is_changed;
}
