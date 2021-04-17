/* header for kmeans++ */
#define ERROR_MSG "Running Kmeans calculation using c module has failed"

int kmeans(PyObject *observations, int k, int n, int d, int max_iter, long* index, long* obs_cluster_array);
