/*
capi.c exposes C API extension for used by kmeans.py. This module makes the required type converts between c and python.
The module triggered by kmeans.py.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.h" //include kmeans header
#include "stdlib.h"
#include "stdio.h"

static PyObject* run (PyObject *self, PyObject *args);

static void convert_index_to_c(PyObject *index, long* index_c);

static PyObject* convert_cluster_to_py(long* obs_cluster_array, int n);

static void convert_index_to_c(PyObject *index, long* index_c){
    /*convert index list to C */
    int i;
    PyObject *ind;
    Py_ssize_t index_num;
    index_num=PyList_Size(index);
    for (i=0; i<index_num; i++){
        ind=PyList_GetItem(index, i);
        if (!PyLong_Check(ind)){
                PyErr_Format(PyExc_ValueError, ERROR_MSG);
                }
        index_c[i]=PyLong_AsLong(ind);
        if (index_c[i]==-1 && PyErr_Occurred()){
            PyErr_Format(PyExc_ValueError, ERROR_MSG);
        }
    }
}

static PyObject* convert_cluster_to_py(long* obs_cluster_array, int n){
    /*convert index list to C */
    int i;
    PyObject *cluster, *clusters_list=PyList_New(n);
    for (i=0; i<n; i++){
        cluster=Py_BuildValue("i", obs_cluster_array[i]);
        PyList_SetItem(clusters_list, i, cluster);
    }
    return clusters_list;
}

static PyObject* run (PyObject *self, PyObject *args){
    /*
    * this function is the module's endpoint for communicating with python script
    * the function calculates kmeans algorithm and prints it to the requested file.
    * args:
        input_observation - matrix of observation where the first k observation are the clusters
        K - number of centroids required
        N - number of observations
        d - the dimension of each observation
        MAX - max iterations the script should do
        index - the origin index of the observation where the k first is the k centroids we start with
    * returns a list of clusters to python if the process ended successfully
    */

    int K,N,d,MAX_ITER;
    PyObject *input_observation, *index, *clusters_list;
    long* index_c, *obs_cluster_array;

    if(!PyArg_ParseTuple(args, "(OiiiiO):run", &input_observation, &K, &N, &d, &MAX_ITER, &index)) {
        return PyErr_Format(PyExc_ValueError,"%s,\nk-means c-api missing arguments",ERROR_MSG);
    }
    if (!PyList_Check(input_observation)){
        return PyErr_Format(PyExc_ValueError, "%s,\ninput_observation should be a list", ERROR_MSG);
    }
     if (!PyList_Check(index)){
        return PyErr_Format(PyExc_ValueError, "%s,\ndata_origin_index should be a list", ERROR_MSG);
    }

    index_c=malloc(N*sizeof(long));
    if (index_c==NULL){
        return PyErr_Format(PyExc_ValueError, "%s,\nmemory allocation failed",ERROR_MSG);
    }
    convert_index_to_c(index, index_c);
    obs_cluster_array=malloc(N*sizeof(long));
    if (obs_cluster_array==NULL){
        free(index_c);
        return PyErr_Format(PyExc_ValueError, "%s,\nmemory allocation failed",ERROR_MSG);
    }
    //run kmeans
    if (kmeans(input_observation, K, N, d, MAX_ITER, index_c, obs_cluster_array)==-1){
        return PyErr_Format(PyExc_ValueError, "%s,\nmemory allocation failed",ERROR_MSG);
    }
    //convert c list to pyobject
    clusters_list=convert_cluster_to_py(obs_cluster_array,N);
    free(obs_cluster_array);
    free(index_c);
    return clusters_list;
}


static PyMethodDef capiMethods[] = {
    {"run",                   /* the Python method name that will be used */
     (PyCFunction)run, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parametersaccepted for this function */
      PyDoc_STR("kmeans++")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};


/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods /* the PyMethodDef array from before containing the methods of the extension */
};


/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}