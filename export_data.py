import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection=projection)
    plot(dimension_number, samples, clusters_kmeans, ax1)
    ax2 = fig.add_subplot(122, projection=projection)
    plot(dimension_number, samples, clusters_spectral, ax2)
    j_k, j_s = jaccard_measure(clusters_kmeans, clusters_spectral, header)
    fig.text(0.5, -0.1, s=set_text(n, k_generated, k_used, j_k, j_s), fontsize=14, ha='center')
    plt.show()
    fig.savefig(r'C:\Users\user\PycharmProjects\final_project\Charts.pdf', bbox_inches='tight')


def plot(dimension_number, samples, clusters, ax):
    """
    scatter plot with colors according to clusters
    """
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2] if dimension_number == 3 else 0
    sctt = ax.scatter(x, y, z, c=clusters)
    return sctt


def set_text(n, k_generate, k_used, j_kmeans, j_spectral):
    """ set plots description text """
    text = "Data was generated from the values:\n" + "n = " + str(n) + ", k =" + str(k_generate) + "\n"
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
    j_kmeans = float(len(header_pairs.intersection(kmeans_pairs)) / len(header_pairs.union(kmeans_pairs)))
    j_spectral = float(len(header_pairs.intersection(spectral_pairs)) / len(header_pairs.union(spectral_pairs)))
    return j_kmeans, j_spectral
#
# samples, header = make_blobs(n_samples=7, centers=3, n_features=3,
#                       random_state=0)
# # print(header)
# # k,s=jaccard_measure([1,1,2,2,0,0,1], [1,1,0,2,2,0,1], header)
# # print(k, s)
# create_pdf_file(samples, header, [1,1,2,2,0,0,1], [1,1,0,2,2,0,1] ,3, 3, 7)
