import numpy as np


def create_affinity_graph(data, scale):
    msq = data * data
    scaled = -msq / scale
    scaled[np.where(np.isnan(scaled))] = 0.0
    aff_matrix = np.exp(scaled)
    aff_matrix.flat[::data.shape[0]+1] = 0.0  # zero out the diagonal
    return aff_matrix