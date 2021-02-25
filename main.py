import argparse
import random

from sklearn.datasets import make_blobs
from k_means import kmeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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
        k = user_k
        n = user_n

    samples, header = make_blobs(n_samples=n, centers=k, n_features=dimension_number,
                                 random_state=0)

    return samples, header

def visualization(samples_kmeans, clusters_kmeans, samples_spectral, clusters_spectral):
    dimension_number=len(samples_kmeans[0])
    projection = '3d' if dimension_number == 3 else None
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection=projection)
    sctt1=plot(dimension_number,samples_kmeans, clusters_kmeans, ax1)
    ax2 = fig.add_subplot(122, projection=projection)
    sctt2=plot(dimension_number,samples_spectral, clusters_spectral, ax2)
    fig.text(0.2, -0.1, s=set_text(100, 10, 10, 1, 1), fontsize=14)
    plt.show()


def plot(dimension_number, samples, clusters, ax):
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2] if dimension_number == 3 else 0
    sctt=ax.scatter(x, y, z, c=clusters)
    return sctt


def set_text(n,k_generate,k_used, j_kmeans, j_spectral):
    text="Data was generated from the values:\n"+ "n = "+str(n) +", k ="+str(k_generate)+"\n"
    text+="The k that was used for both algorithms was "+ str(k_used)+"\n"
    text+="The Jaccard measure for Spectral Clustering: "+str(j_spectral)+ "\n"
    text+="The Jaccard measure for K-means: "+str(j_kmeans)
    return text

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
    # kmeans(k, n, d, MAX_ITER, samples)
