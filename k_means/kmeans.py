import pandas as pd
import numpy as np
import mykmeanssp as km


def k_mean(K, N, d, MAX_ITER, observations_matrix):
    """
    This function crates arranges the data to be:
        K clusters would be the k first lines at the matrix
        next N-K observations would be none clustered observation
    eventually the function calculates Kmeans algorithm using c module
    args:
        K - number of centroids required
        N - number of observations
        d - the dimension of each observation
        MAX - max iterations the script should do
        path - path to the data
    """
    np.random.seed(0)
    centroid_index_arr = np.empty(K, int)
    centroids_matrix = create_k_clusters(observations_matrix, N, K, d, centroid_index_arr)
    ind = [i for i in range(N) if i not in centroid_index_arr]
    data_origin_index=(np.concatenate((centroid_index_arr, ind))).tolist()
    observations_matrix = (np.concatenate((centroids_matrix, observations_matrix[ind]), axis=0)).tolist()
    return km.run([observations_matrix, K, N, d, MAX_ITER, data_origin_index])


def create_k_clusters(observations_matrix, N, K, d, centroid_index_arr):
    centroids_matrix = np.zeros((K, d), dtype=np.float64)
    first_centroid_index = np.random.choice(N, 1)
    centroids_matrix[0] = observations_matrix[first_centroid_index[0]]
    centroid_index_arr[0] = first_centroid_index
    find_next_centroids(observations_matrix, centroids_matrix, K, N, centroid_index_arr)
    return centroids_matrix


def find_next_centroids(observations_matrix, centroids_matrix, K, N, centroid_index_arr):
    i = 1  # already found one above

    distance_matrix = np.zeros((K, N))
    distance_matrix[0]=squared_euclidean_distance(observations_matrix, centroids_matrix[0])

    while (i < K):
        # Run until we find k centroids
        min_d_arr=np.min(distance_matrix[:i,], axis=0)
        min_d_arr = min_d_arr / (min_d_arr.sum())
        next_centroid_index = np.random.choice(N, 1, p=min_d_arr)
        centroid_index_arr[i] = next_centroid_index
        centroids_matrix[i] = observations_matrix[next_centroid_index]
        distance_matrix[i]=squared_euclidean_distance(observations_matrix, centroids_matrix[i])
        i += 1


def squared_euclidean_distance(observation, centroids_df):
    """find cluster’s centroid using squared Euclidean distance
    observation and centroid are lists of size D"""
    dist = (np.power((observation-centroids_df), 2)).sum(axis=1)

    return dist