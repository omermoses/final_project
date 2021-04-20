"""
export_data.py creates the project's required files- data.txt, clusters.txt and Charts.pdf.
The module uses matplotlib and numpy to provide his service.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_pdf_file(samples, header, clusters_kmeans, clusters_spectral, k_generated, k_used, n):
    """
    create visualization output and cluster+data pdfs
    params: samples, header- data from make_bolbs
            clusters_kmeans- for sample i, clusters_kmeans[i]==the cluster that sample i belongs,
                            to according to kmeans
            clusters_spectral- same for spectral
            k_generated- the k that we used to generate the data with make_bolbs
            k_used- the k that we used to cluster with kmeans++ and spectral
            n- n umber of samples
    return- PDF file with the plots and some data, pdf for clustera and pdf for data
    """

    # plots
    dimension_number = len(samples[0])
    projection = '3d' if dimension_number == 3 else None
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(121, projection=projection)
    sctt1 = plot(dimension_number, samples, clusters_spectral, ax1, "Normalized Spectral")
    ax2 = fig.add_subplot(122, projection=projection)
    sctt2 = plot(dimension_number, samples, clusters_kmeans, ax2, "k_means")
    j_k, j_s = jaccard_measure(clusters_kmeans, clusters_spectral, header)
    if dimension_number == 2:
        fig.text(0.485, -0.2, s=set_text(n, k_generated, k_used, j_k, j_s), fontsize=14, ha='center')
    else:
        fig.text(0.525, 0, s=set_text(n, k_generated, k_used, j_k, j_s), fontsize=14, ha='center')
    plt.show()
    fig.savefig(r'Charts.pdf', bbox_inches='tight')

    # data file
    write_data(samples, header, n)

    # clusters file
    write_clusters(clusters_kmeans, clusters_spectral, k_used)



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
    j_kmeans = calculate_j(header_pairs, kmeans_pairs)
    j_spectral = calculate_j(header_pairs, spectral_pairs)
    return j_kmeans, j_spectral


def calculate_j(set_1, set_2):
    intersect = 0
    for pair in set_1:
        if pair in set_2:
            intersect += 1
    return intersect / (len(set_1) + len(set_2) - intersect)


def write_clusters(clusters_kmeans, clusters_spectral, k_used):
    """
    writes the number of clusters-k used,
    the clusters for the data using Normalized Spectral Clustering,
    and the clusters for the data using k-means++
    into clusters.txt file.
    """
    with open("clusters.txt", 'w') as file:
        file.write(str(k_used) + '\n')
        for i in range(k_used):
            file.write(','.join(map(str, np.argwhere(clusters_spectral == i).flatten())) + '\n')
        for i in range(k_used-1):
            file.write(','.join(map(str, np.argwhere(clusters_kmeans == i).flatten())) + '\n')
        file.write(','.join(map(str, np.argwhere(clusters_kmeans == k_used-1).flatten())))


def write_data(observations, clusters, n):
    """
    writes the data from make_blobs with 8 digits after dot, and each point cluster by make_blobs into
    data.txt file.
    """
    with open("data.txt", 'w') as file:
        for i in range(n-1):
            file.write(','.join(map("{:.8f}".format, observations[i])) + ',' + str(clusters[i]) + '\n')
        file.write(','.join(map("{:.8f}".format, observations[n-1])) + ',' + str(clusters[n-1]))

