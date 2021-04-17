"""
Main.py receives the arguments from the user, generates data and starts the execution of Spectral Clustering and k means
The module uses capi.c, kmeans.py and export data to provide his service.
"""
import argparse
import random
import time
import numpy as np

import s_clustring

import kmeans
from sklearn.datasets import make_blobs
import export_data

MAXIMUM_CAPACITY_2 = [540, 30]
MAXIMUM_CAPACITY_3 = [535, 30]
MAX_ITER = 300


def handle_samples(user_k, user_n, is_random):
    """
        this function creates the samples for the algorithm
    :return:
        samples - The generated samples (ndarray type)
        header - The integer labels for cluster membership of each sample (ndarray type)
    """

    dimension_number =random.randint(2, 3)

    if is_random:
        # The Random flag is true so we choose randomly k and n values
        if dimension_number == 2:
            k = random.randint(MAXIMUM_CAPACITY_2[1] // 2, MAXIMUM_CAPACITY_2[1])
            n = random.randint(MAXIMUM_CAPACITY_2[0] // 2, MAXIMUM_CAPACITY_2[0])
        elif dimension_number == 3:
            k = random.randint(MAXIMUM_CAPACITY_3[1] // 2, MAXIMUM_CAPACITY_3[1])
            n = random.randint(MAXIMUM_CAPACITY_3[0] // 2, MAXIMUM_CAPACITY_3[0])
    else:
        # Random==False so we use the k,n of the user
        k = user_k
        n = user_n
        if k <= 0 or n <= 0:
            # if the user didn't entered values for k/n, the default is -1 and it's an error
            # if the users k/n is <=0 it's an error
            # print("parameters should be greater then 0")
            print("parameters are missing or incorrect")
            exit(1)

        elif k >= n:
            print("K should be smaller then N")
            exit(1)

    samples, header = make_blobs(n_samples=n, centers=k, n_features=dimension_number)
    return samples, header, k, n, dimension_number


if __name__ == '__main__':
    """
        • k - number of clusters
        • n - number of data points
        • Random - indicates the way the data is to be generated
    """
    start = time.time()  ####

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('k', action='store', type=int)
    my_parser.add_argument('n', action='store', type=int)
    my_parser.add_argument('--Random', default=True, action='store_false', help='Bool type')

    args = my_parser.parse_args()

    #print max capacity
    print("Maximum capacity of 2 dimension points is N={} K={}".format(MAXIMUM_CAPACITY_2[0], MAXIMUM_CAPACITY_2[1]))
    print("Maximum capacity of 3 dimension points is N={} K={}".format(MAXIMUM_CAPACITY_3[0], MAXIMUM_CAPACITY_3[1]))

    # Generate data for the algorithms
    samples, header, k_generated, n, d = handle_samples(args.k, args.n, args.Random)

    # Execute Normalized Spectral Clustering Comparison
    if args.Random:
        # k is determine by the eigengap_heuristic
        spectral_data, k_used = s_clustring.spectral_clustering(samples, n, None)
    else:
        # use the users k, which is equal to k_generated
        spectral_data, k_used = s_clustring.spectral_clustering(samples, n, k_generated)
    spectral_clusters = np.array(kmeans.k_mean(k_used, n, k_used, MAX_ITER, spectral_data))

    # Execute K-means algorithm
    kmeans_clusters = np.array(kmeans.k_mean(k_used, n, d, MAX_ITER, samples))

    # Export data
    export_data.create_pdf_file(samples, header, kmeans_clusters, spectral_clusters, k_generated, k_used, n)
    end = time.time()  ####
    print("--- %s seconds ---" % (end - start))  ####
