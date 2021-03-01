import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_pdf_file(samples_kmeans, clusters_kmeans, samples_spectral,clusters_spectral ,k_generated, k_used, n):
    with PdfPages(r'C:\Users\User\PycharmProjects\final_project\final_project\Charts.pdf') as export_pdf:
        dimension_number = len(samples_kmeans[0])
        projection = '3d' if dimension_number == 3 else None
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection=projection)
        sctt1 = plot(dimension_number, samples_kmeans, clusters_kmeans, ax1)
        ax2 = fig.add_subplot(122, projection=projection)
        sctt2 = plot(dimension_number, samples_spectral, clusters_spectral, ax2)
        fig.text(0.2, -0.1, s=set_text(n, k_generated, k_used, j_kmeans,j_spectral), fontsize=14)
        plt.show()


def plot(dimension_number, samples, clusters, ax):
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2] if dimension_number == 3 else 0
    sctt = ax.scatter(x, y, z, c=clusters)
    return sctt

def set_text(n, k_generate, k_used, j_kmeans, j_spectral):
    text = "Data was generated from the values:\n" + "n = " + str(n) + ", k =" + str(k_generate) + "\n"
    text += "The k that was used for both algorithms was " + str(k_used) + "\n"
    text += "The Jaccard measure for Spectral Clustering: " + str(j_spectral) + "\n"
    text += "The Jaccard measure for K-means: " + str(j_kmeans)
    return text

def jaccard_measure(clusters, origin_header):

