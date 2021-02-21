import argparse
import random

from sklearn.datasets import make_blobs
from k_means import kmeans


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
        k = random.randint(K_MAXIMUM_CAPACITY//2, K_MAXIMUM_CAPACITY)
        n = random.randint(N_MAXIMUM_CAPACITY//2, N_MAXIMUM_CAPACITY)
    else:
        k = user_k
        n = user_n

    samples, header = make_blobs(n_samples=n, centers=k, n_features=dimension_number,
                      random_state=0)

    return samples, header



if __name__ == '__main__':
    """
        • k - number of clusters
        • n - number of data points
        • Random - indicates the way the data is to be generated
    """

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('k', action='store', type=int)
    my_parser.add_argument('n', action='store', type=int)
    my_parser.add_argument('Random', action='store', default=True, type=bool)

    args = my_parser.parse_args()
    if args.k <= 0 or args.n <= 0:
        print("parameters should be greater then 0")
        exit(1)

    elif args.k >= args.n:
        print("K should be smaller then N")
        exit(1)

    # Generate data for the algorithms
    samples, headers = handle_samples(args.k, args.n, args.Random)

    # Execute Normalized Spectral Clustering Comparison

    # Execute K-means algorithm
    #kmeans(k, n, d, MAX_ITER, samples)



