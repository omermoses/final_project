import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.datasets import make_blobs



def create_pdf_file(samples, header, clusters_kmeans, clusters_spectral, k_generated, k_used, n):
    """
    create visualization output
    params: samples, header- data from make_bolbs
            clusters_kmeans- for sample i, clusters_kmeans[i]==the cluster that sample i belongs,
                            to according to kmeans
            clusters_spectral- same for spectral
            k_generated- the k that we used to generate the data with make_bolbs
            k_used- the k that we used to cluster with kmeans++ and spectral
            n- n umber of samples
    return- PDF file with the plots and some data
    """
    dimension_number = len(samples[0])
    projection = '3d' if dimension_number == 3 else None
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(121, projection=projection)
    sctt1 = plot(dimension_number, samples, clusters_kmeans, ax1, "k_means")
    ax2 = fig.add_subplot(122, projection=projection)
    sctt2 = plot(dimension_number, samples, clusters_spectral, ax2, "Normalized Spectral")
    j_k, j_s = jaccard_measure(clusters_kmeans, clusters_spectral, header)
    fig.text(0.5, -0.2, s=set_text(n, k_generated, k_used, j_k, j_s), fontsize=14, ha='center')
    # write_clusters(clusters_kmeans, clusters_spectral, k_generated, k_used)
    plt.show()
    fig.savefig(r'Charts.pdf', bbox_inches='tight')

    # for tests
    fig_1 = plt.figure(dpi=500)
    ax3 = fig_1.add_subplot(121, projection=projection)
    sctt3 = plot(dimension_number, samples, header, ax3, "make blobs")
    plt.show()
    fig_1.savefig(r'make_blobs.pdf', bbox_inches='tight')



def plot(dimension_number, samples, clusters, ax, title):
    """
    scatter plot with colors according to clusters
    """
    x = samples[:, 0]
    y = samples[:, 1]
    if dimension_number == 3:
        z = samples[:, 2] if dimension_number == 3 else 0
        sctt = ax.scatter(x, y, z, c=clusters)

    else:
        sctt = ax.scatter(x, y, c=clusters)
    plt.title(title)
    return sctt


def set_text(n, k_generate, k_used, j_kmeans, j_spectral):
    """ set plots description text """
    text = "Data was generated from the values:\n" + "n = " + str(n) + ", k = " + str(k_generate) + "\n"
    text += "The k that was used for both algorithms was " + str(k_used) + "\n"
    text += "The Jaccard measure for Spectral Clustering: " + str("{:.2f}".format(j_spectral)) + "\n"
    text += "The Jaccard measure for K-means: " + str("{:.2f}".format(j_kmeans))
    return text


def jaccard_measure(clusters_kmeans, clusters_spectral, origin_header):
    """
    calculate Jaccard measure for kmeans++ and spectral clustering
    params: clusters_kmeans- for sample i, clusters_kmeans[i]==the cluster that sample i belongs,
                            to according to kmeans
            clusters_spectral- same for spectral
            origin_header- header from make_blobs
    return: Jaccard measure for kmeans, spectral
    """
    header_pairs = set([(i, j) for i in range(len(origin_header)) for j in range(i + 1, len(origin_header)) if
                        origin_header[i] == origin_header[j]])
    kmeans_pairs = set([(i, j) for i in range(len(clusters_kmeans)) for j in range(i + 1, len(clusters_kmeans)) if
                        clusters_kmeans[i] == clusters_kmeans[j]])
    spectral_pairs = set([(i, j) for i in range(len(clusters_spectral)) for j in range(i + 1, len(clusters_spectral)) if
                          clusters_spectral[i] == clusters_spectral[j]])
    j_kmeans=calculate_j(header_pairs, kmeans_pairs)
    j_spectral=calculate_j(header_pairs, spectral_pairs)
    # j_kmeans = float(len(header_pairs.intersection(kmeans_pairs)) / len(header_pairs.union(kmeans_pairs)))
    # j_spectral = float(len(header_pairs.intersection(spectral_pairs)) / len(header_pairs.union(spectral_pairs)))
    return j_kmeans, j_spectral

def calculate_j(set_1, set_2):
    intersect = 0
    for pair in set_1:
        if pair in set_2:
            intersect += 1
    return intersect / (len(set_1) + len(set_2) - intersect)


def write_clusters(clusters_kmeans, clusters_spectral, k_generated, k_used):

    with open("clusters.txt", 'w') as file:
        file.write(str(k_used) + '\n')
        for i in range(k_used):
            file.write(','.join(map(str, np.argwhere(clusters_spectral == i).flatten())) + '\n')
        for i in range(k_used):
            file.write(','.join(map(str, np.argwhere(clusters_kmeans == i).flatten())) + '\n')


# samples, header = make_blobs(n_samples=7, centers=3, n_features=2,
#                        random_state=0)
# # # # # # print(header)
# # k,s=jaccard_measure([1,1,2,2,0,1,1], [1,1,0,2,2,0,1], header)
# # # # # # print(k, s)
# create_pdf_file(samples, header, [1,1,2,2,0,0,1], [1,1,0,2,2,0,1] ,3, 3, 7)
