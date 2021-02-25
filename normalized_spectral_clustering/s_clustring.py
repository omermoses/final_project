import numpy as np
from sklearn.datasets import make_blobs


def spectral_clustering(data, n):
    # Calculate weighted adjacency matrix
    W = create_weighted_adjacency_matrix(data, n)

    # Calculate L norm
    LN = calculate_L_norm(W, n)

    # Find eigenvalues and eigenvectors
    A, Q = qr_algorithm(LN, n)

    # Sorted eigenvalues and eigenvectors
    sorted_eigenvalues_index = np.argsort(np.diagonal(A))
    sorted_eigenvalues = np.diagonal(A)[sorted_eigenvalues_index]
    sorted_eigenvectors = Q[:, sorted_eigenvalues_index]

    # Determine k
    k = eigengap_heuristic(sorted_eigenvalues, n)  #(int)
    print(k)

    # Create U
    print(sorted_eigenvectors)
    sorted_eigenvectors = sorted_eigenvectors[:, :k+1]

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
        col = data - data[j, :]
        col = np.linalg.norm(col, 2, axis=1)
        col = np.exp(-col / 2)
        W[:, j] = col
    np.fill_diagonal(W, val=0.0)  # zero out the diagonal
    return W


def calculate_L_norm(W, n):
    """
    create L normalized
    params: wighted matrix
            n - number of samples
        return: LN - The normalized L

    """
    I = np.identity(n, dtype='float64')
    D_times_half = np.diag(np.power(W.sum(axis=1, dtype='float64'), -0.5))
    return I - (np.matmul(np.matmul(D_times_half, W), D_times_half))


def create_diagonal_degree_matrix(wighted_matrix):
    """
    create D
    params: wighted matrix
            n- number of samples
        return: D - diagonal degree matrix

    """
    return np.diag(wighted_matrix.sum(axis=1, dtype='float64'))


# A = np.asarray([[3, 2, 4], [2, 0, 2], [4, 2, 3]], dtype='float64')
# w = create_weighted_adjacency_matrix(A, 3)
# print(w)


def gram_schmidt(A, n):
    """
        calculate Modified Gram Schmidt
        params: A- ndarray of size nXn with dtype='float64'
                n- A dim
        return: Q- orthogonal matrix
                R- diagonal matrix
    """

    U = A.copy()
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i] = np.linalg.norm(U[:, i], 2)  # L2 norm
        Q[:, i] = U[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = (Q[:, i]).T @ U[:, j]
            U[:, j] = U[:, j] - R[i, j] * Q[:, i]
    return Q, R


def qr_algorithm(A, n):
    """
    e = 0.0001
    calculate eigenvalues and eigenvectors
    params: A- ndarray of size nXn with dtype='float64'
            n- A dim
    return: a- matrix with the eigenvalues on the diagonal
            q- eigenvectors as columns

    """
    e = 0.0001
    a = A
    q = np.identity(n)
    for i in range(n):
        Q, R = gram_schmidt(a, n)
        a = R @ Q
        dist = np.absolute(q) - np.absolute(q @ Q)
        if -e <= dist.all() <= e:
            break
        q = q @ Q
    return a, q


def eigengap_heuristic(eigenvals_array, n):
    """
       determine k
       params: eigenvals_array- ***sorted*** eigenvalues
               n- number of samples
       return: k- argmax(delta_i) when i from 0...n/2-1
       """
    gaps = np.abs(eigenvals_array[1:int(n / 2)] - eigenvals_array[:int(n / 2) - 1])
    k = np.argmax(gaps)  # returns the smallest index in case of equility
    return k


def create_t_matrix(U, n):
    """
    Form matrix T from U by renormalizing each of U’s rows to have unit length,

    :return:
    """
    print(U)
    print((np.power(np.sum(np.power(U, 2), axis=1), 0.5)).reshape(n, 1))
    return U/(np.power(np.sum(np.power(U, 2), axis=1), 0.5)).reshape(n, 1)





# A = np.asarray([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype='float64')
# A = sklearn.datasets.make_spd_matrix(10)
# print("A\n" , A)
# print("***************")
#
Y = np.array([[5,3,1,4],[3,6,0,2.5],[1,0,3,1.7],[4,2.5, 1.7, 10]])
# Q,R= gram_schmidt(A,10)
# print("Q\n", Q)
# print("***************")
# print("R\n",R)
# print("**************")
# print("Q.T @ Q\n",Q.T @ Q)
# print("**************")
# print("Q @ R\n",Q @ R)
# print("**************")

# a, q = qr_algorithm(Y, 4)
# print("a diagonal\n", np.diagonal(a))
# print("*************")
# print("q\n", q)
# print(np.linalg.eig(A))
# sorted_eigenvalues_index=np.argsort(np.diagonal(a))
# sorted_eigenvalues=np.diagonal(a)[sorted_eigenvalues_index]
# sotred_eigenvectors=q[:,sorted_eigenvalues_index]
# print(sorted_eigenvalues_index)
# print(sorted_eigenvalues)
# print(sotred_eigenvectors)

# print(eigengap_heuristic(np.asarray([1, 2, 3, 4, 5, 6, 7, 8]), 8))

#
samples, header = make_blobs(n_samples=10, centers=5, n_features=3,
                      random_state=0)
spectral_clustering(Y, 4)
