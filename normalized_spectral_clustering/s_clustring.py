import numpy as np
from sklearn.datasets import make_blobs

EPS = 0.0001



def spectral_clustering(data, n, k):
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

    if (k is None):
        # Determine k, else - use the k from function params
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
    # W = np.zeros((n, n), order='F')
    W = np.zeros((n, n))

    for j in range(n):
        col = np.linalg.norm(data - data[j, :], 2, axis=1)
        col = np.exp(-col / 2)
        W[:, j] = col
    np.fill_diagonal(W, val=0.0)  # zero out the diagonal
    # np.einsum('ii->i', W)[:] = 0  # zero out the diagonal
    return W


def calculate_L_norm(W, n):
    """
    create L normalized
    params: wighted matrix
            n - number of samples
        return: LN - The normalized L

    """
    # I = np.identity(n, dtype='float64')
    # D_times_half = np.diag(np.power(np.sum(W, axis=1, dtype='float64'), -0.5))
    D_times_half = np.diag(np.power(np.einsum('ij->i', W), -0.5))
    return np.identity(n, dtype='float64') - (np.matmul(np.matmul(D_times_half, W), D_times_half))
    # return np.identity(n, dtype='float64') - np.einsum('ij,jk', np.einsum('ij,jk', D_times_half, W), D_times_half)

# def gram_schmidt(A, n):
#     """
#         old
#     """
#
#     U = A.copy(order='F')
#     Q = np.zeros((n, n), order='F')
#     R = np.zeros((n, n))
#     for i in range(n):
#         Ui = U[:, i]
#         norm = np.linalg.norm(Ui, 2)  # L2 norm
#         R[i, i] = norm
#         Qi = Q[:, i]
#         for j in range(i + 1, n):
#             Uj = U[:, j]
#             R[i, j] = Qi @ Uj
#             # U[:, j] -= R[i, j] * Q[:, i]
#             U[:, j] = Uj - R[i, j] * Qi
#     return Q, R


def gram_schmidt(A, n):
    """
            calculate Modified Gram Schmidt
            params: A- ndarray of size nXn with dtype='float64'
                    n- A dim
            return: Q- orthogonal matrix
                    R- diagonal matrix
        """
    # U = A.copy(order='F')
    # Q = np.zeros((n, n), order='F')
    U = A.copy()
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    for i in range(n):
        U_i = U[:, i]
        norm = np.linalg.norm(U_i, 2)  # L2 norm
        R[i, i] = norm
        # raise an error??
        if norm == 0:  # avoid dividing by 0
            Q_i = Q[:, i] = np.zeros(n)
        else:
            Q_i = Q[:, i] = U_i / norm
        # U_j_cols = U[:, i + 1:n]
        R[i, i + 1:n] = np.einsum('i,ij->j', Q_i, U[:, i + 1:n])
        # R[i, i + 1:n] = R_row_i = np.einsum('i,ij->j', Q_i, U_j_cols)
        U[:, i + 1:n] -= np.einsum('i,j->ji', R[i, i + 1:n], Q_i)
        # U[:, i + 1:n] = U_j_cols - np.einsum('i,j->ji', R_row_i, Q_i)

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
    a = A
    q = np.identity(n, dtype=np.float64)
    for i in range(n):
        Q, R = gram_schmidt(a, n)
        # a = np.einsum('ij,jk', R, Q)
        a = R @ Q
        # q_temp = np.einsum('ij,jk', q, Q)
        q_temp = q @ Q
        dist = np.absolute(q) - np.absolute(q_temp)
        # if (-e <= dist).all() and (dist <= e).all():
        if (np.absolute(dist) <= EPS).all():
            break
        q = q_temp
    return a, q


def eigengap_heuristic(eigenvals_array, n):
    """
       determine k
       params: eigenvals_array- ***sorted*** eigenvalues
               n- number of samples
       return: k- argmax(delta_i) when i from 0...n/2-1
       """
    i = int(np.ceil(n / 2))  # celling(n/2)
    gaps = np.abs(eigenvals_array[1:i + 1] - eigenvals_array[:i])
    k = np.argmax(gaps)  # returns the smallest index in case of equility
    return k + 1

def create_t_matrix(U, n):
    """
    Form matrix T from U by renormalizing each of U’s rows to have unit length,

    :return:
    """
    return U / (np.power(np.sum(np.power(U, 2), axis=1), 0.5)).reshape(n, 1)

# """tests"""
#
# def test():
#     # data = np.array([[4, 9], [2, 6]])
#     n = 10
#     data, header = make_blobs(n_samples=n, centers=4, n_features=3,
#                                                               random_state=0)
#     data_points = data
#     weighted_adjacency_matrix = create_weighted_adjacency_matrix(data_points, n)
#     # diagonal_degree_matrix = create_diagonal_degree_mat(weighted_adjacency_matrix)
#     L_norm = calculate_L_norm(weighted_adjacency_matrix, n)
#     A_bar, Q_bar = qr_algorithm(L_norm, n)
#     sorted_eigenvalues_index = np.argsort(np.diagonal(A_bar))
#     sorted_eigenvalues = np.diagonal(A_bar)[sorted_eigenvalues_index]
#     sorted_eigenvectors = Q_bar[:, sorted_eigenvalues_index]
#     # Create U
#     sorted_eigenvectors = sorted_eigenvectors[:, :4]
#
#     # Creat T matrix
#     T = create_t_matrix(sorted_eigenvectors, n)
#
#
#     k = eigengap_heuristic(sorted_eigenvalues, n)  # (int)
#     w, v = np.linalg.eig(L_norm)
#     print(f"my eig are {np.diag(A_bar)}\n")
#     print(f"np eig are {w}\n")
#     print(f"my vec are {(Q_bar)}\n")
#     print(f"np vec are {v}\n")
#
#
# test()


# def test_k_value(n, K, D,std):
#     data_points, points_membership = make_blobs(n_samples=n, centers=K, n_features=D,cluster_std=std, center_box=(-3.0,3))
#     weighted_adjacency_matrix = create_weighted_adjacency_matrix(data_points,n)
#     # diagonal_degree_matrix = create_diagonal_degree_mat(weighted_adjacency_matrix)
#     L_norm = calculate_L_norm(weighted_adjacency_matrix, n)
#     A_bar, Q_bar = qr_algorithm(L_norm, n)
#
#     sorted_eigenvalues_index = np.argsort(np.diagonal(A_bar))
#     sorted_eigenvalues = np.diagonal(A_bar)[sorted_eigenvalues_index]
#     #
#     #
#     our_k = eigengap_heuristic(sorted_eigenvalues, n)  # (int)
#     data_k = K
#     return 1 if np.abs(our_k- data_k) <=1 else 0
#
# def test_k_correlation():
#     num_of_eq = 0
#     num_of_runs = 2
#     std = 1
#     for i in range(num_of_runs):
#         n = np.random.randint(10,30)
#         K = np.random.randint(3,20)
#         num_of_eq+=test_k_value(n, K, 3,std)
#     print(num_of_eq/num_of_runs)
#
# test_k_correlation()