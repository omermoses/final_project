"""
    s_clustering.py calculate the number of clusters-k and the new data matrix-T using Normalized Spectral Clustering.
    this data and k will be provided to k-means++ algorithm for the clustering.

    Normalized Spectral Clustering uses the original samples from make_blobs-n samples with 2/3 dims, and returns a new
    data matrix with size nXk, which will be used for clustering using k-means++.
    the main function- spectral_clustering,
    help functions- create_weighted_adjacency_matrix
                    calculate_L_norm
                    gram_schmidt
                    qr_algorithm
                    eigengap_heuristic
                    create_t_matrix
"""

import numpy as np

EPS = 0.0001

def spectral_clustering(data, n, k):
    """
    params: data- ndarray with the samples in the rows
            n- number of samples
            k- int if Random==False, else None
    return: k- int-original k or by eigengap_heuristic
            T- the new data matrix as ndarray
    """
    # Calculate weighted adjacency matrix
    W = create_weighted_adjacency_matrix(data, n)

    # Calculate L norm
    LN = calculate_L_norm(W, n)

    # Find eigenvalues and eigenvectors
    A, Q = qr_algorithm(LN, n)

    # Sort eigenvalues and eigenvectors
    sorted_eigenvalues_index = np.argsort(np.diagonal(A))
    sorted_eigenvalues = np.diagonal(A)[sorted_eigenvalues_index]
    sorted_eigenvectors = Q[:, sorted_eigenvalues_index]

    if (k is None):
        # Determine k if Random==True, else - use the k from function params
        k = eigengap_heuristic(sorted_eigenvalues, n)  # (int)

    # Create U
    sorted_eigenvectors = sorted_eigenvectors[:, :k]

    # Creat T matrix
    T = create_t_matrix(sorted_eigenvectors, n)

    return T, k


def create_weighted_adjacency_matrix(data, n):
    """
    create W
    params: data- ndarray with the samples in the rows
            n- number of samples
    return: W- weighted_adjacency_matrix
    """
    W = np.zeros((n, n))

    for j in range(n):
        col = np.linalg.norm(data - data[j, :], 2, axis=1)
        col = np.exp(-col / 2)
        W[:, j] = col
    np.fill_diagonal(W, val=0.0)
    return W


def calculate_L_norm(W, n):
    """
    create L normalized
    params: wighted matrix
            n - number of samples
        return: LN - The normalized L

    """
    D_times_half = np.diag(np.power(np.sum(W, axis=1, dtype='float64'), -0.5))
    # create L using D_times_0.5
    return np.identity(n, dtype='float64') - (np.dot(np.dot(D_times_half, W), D_times_half))


def gram_schmidt(A, n):
    """
    calculate Q,R using Modified Gram Schmidt
    params: A- ndarray of size nXn with dtype='float64'
            n- A dim
    return: Q- orthogonal matrix
            R- diagonal matrix
    """
    U = A.copy()
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    for i in range(n):
        U_i = U[:, i]
        norm = np.linalg.norm(U_i, 2)
        R[i, i] = norm
        if norm == 0:
            Q_i = Q[:, i] = np.zeros(n)
        else:
            Q_i = Q[:, i] = U_i / norm
        R[i, i + 1:n] = np.einsum('i,ij->j', Q_i, U[:, i + 1:n])
        U[:, i + 1:n] -= np.einsum('i,j->ji', R[i, i + 1:n], Q_i)
    return Q, R


def qr_algorithm(A, n):
    """
    calculate eigenvalues (a diagonal) and eigenvectors (q)
    params: A- ndarray of size nXn with dtype='float64'
            n- A dim
    return: a- matrix with the eigenvalues on the diagonal
            q- eigenvectors as columns
    """
    a = A
    q = np.identity(n, dtype=np.float64)
    for i in range(n):
        Q, R = gram_schmidt(a, n)
        a = R @ Q
        q_temp = q @ Q
        dist = np.absolute(q) - np.absolute(q_temp)
        if (np.absolute(dist) <= EPS).all():
            break
        q = q_temp
    return a, q


def eigengap_heuristic(eigenvals_array, n):
    """
    determine k
    params: eigenvals_array- sorted
            n- number of samples
    return: k- argmax(delta_i) when i from 0...(n/2)-1
    """
    i = int(np.ceil(n / 2))  # celling(n/2)
    gaps = np.abs(eigenvals_array[1:i + 1] - eigenvals_array[:i])
    k = np.argmax(gaps)  # returns the smallest index in case of equility
    return k + 1


def create_t_matrix(U, n):
    """
    Form matrix T from U
    params: U-sorted_eigenvectors
            n- number of samples
    return: T-new data matrix- shape nXk
    """
    return U / (np.power(np.sum(np.power(U, 2), axis=1), 0.5)).reshape(n, 1)
