import argparse
import random

from normalized_spectral_clustering import s_clustring

from k_means import kmeans
from sklearn.datasets import make_blobs
import export_data

K_MAXIMUM_CAPACITY = 10
N_MAXIMUM_CAPACITY = 200
MAX_ITER = 300


def handle_samples(user_k, user_n, is_random):
    """
        this function creates the samples for the algorithm
    :return:
        samples - The generated samples (ndarray type)
        header - The integer labels for cluster membership of each sample (ndarray type)
    """

    dimension_number = random.randint(2, 3)

    if is_random:
        # The Random flag is true so we choose randomly k and n values
        k = random.randint(K_MAXIMUM_CAPACITY // 2, K_MAXIMUM_CAPACITY)
        n = random.randint(N_MAXIMUM_CAPACITY // 2, N_MAXIMUM_CAPACITY)
    else:
        #Random==False so we use the k,n of the user
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


    samples, header = make_blobs(n_samples=n, centers=k, n_features=dimension_number,
                                 random_state=0)

    return samples, header, k, n, dimension_number


if __name__ == '__main__':
    """
        • k - number of clusters
        • n - number of data points
        • Random - indicates the way the data is to be generated
    """

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('k', action='store', type=int)
    my_parser.add_argument('n', action='store', type=int)
    my_parser.add_argument('--Random', default=True, action='store_false', help='Bool type')

    args = my_parser.parse_args()

    # Generate data for the algorithms
    samples, header, k_generated, n, d = handle_samples(args.k, args.n, args.Random)


    # Execute Normalized Spectral Clustering Comparison
    if args.Random:
        # k is determine by the eigengap_heuristic
        spectral_data, k_used = s_clustring.spectral_clustering(samples, n, None)
    else:
        # use the users k, which is equal to k_generated
        spectral_data, k_used = s_clustring.spectral_clustering(samples, n, k_generated)
    spectral_clusters = kmeans.k_mean(k_used, n, k_used, MAX_ITER, spectral_data)


    # Execute K-means algorithm
    kmeans_clusters = kmeans.k_mean(k_used, n, d, MAX_ITER, samples)

    # Export data
    export_data.create_pdf_file(samples, header, kmeans_clusters, spectral_clusters, k_generated, k_used, n)
