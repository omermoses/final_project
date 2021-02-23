import numpy as np


# def create_affinity_graph(data, scale):
#     msq = data * data
#     scaled = -msq / scale
#     scaled[np.where(np.isnan(scaled))] = 0.0
#     aff_matrix = np.exp(scaled)
#     aff_matrix.flat[::data.shape[0] + 1] = 0.0  # zero out the diagonal
#     return aff_matrix


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
    np.fill_diagonal(W, val=0.0) # zero out the diagonal
    return W


A = np.asarray([[3, 2, 4], [2, 0, 2], [4, 2, 3]], dtype='float64')
w = create_weighted_adjacency_matrix(A, 3)
print(w)


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


def qr(A, n):
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
    for i in range(1000):
        Q, R = gram_schmidt(a, n)
        a = R @ Q
        dist = np.absolute(q) - np.absolute(q @ Q)
        if -e <= dist.all() <= e:
            break
        q = q @ Q
    return a, q

A=np.asarray([[2,-1,0],[-1,2,-1],[0,-1,2]], dtype='float64')
# Q,R= gram_schmidt(A,3)
# print(Q)
# print("***************")
# print(R)
# print("**************")
# print(Q.T @ Q)
# print("**************")
# print(Q @ R)

# a,q=qr(A,3)
# print(a)
# print("*************")
# print(q)
# print(np.linalg.eig(A))