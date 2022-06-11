import numpy as np
import scipy.sparse as sp
import metis
from .privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from .sparsegraph import load_from_npz


def sparse_feeder(M):
    # Convert a sparse matrix to the format suitable for feeding as a tf.SparseTensor
    M = M.tocoo()
    return np.vstack((M.row, M.col)).T, M.data, M.shape


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        return sp.csr_matrix((data, indices, indptr), shape=shape)


def split_random(n, n_train):
    rnd = np.random.permutation(n)
    train_idx = np.sort(rnd[:n_train])
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_idx))
    return train_idx, test_idx


def get_data(dataset_path, privacy_amplify_sampling_rate):
    g = load_from_npz(dataset_path)

    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()

    # number of nodes and attributes
    n, d = g.attr_matrix.shape
    class_number = len(np.unique(g.labels))
    print("Loading {} graph with #nodes={}, #attributes={}, #classes={}".format(dataset_path.split('/')[-1], n, d,
                                                                                class_number))

    attr_matrix = g.attr_matrix

    # Generate Train Subgraph and Test Subgraph
    dense_adj_matrix = g.adj_matrix.toarray()
    dense_attr_matrix = attr_matrix.toarray()

    # split train/test graph
    train_adj_lists = [[] for _ in range(len(dense_adj_matrix))]
    for node_index in range(len(dense_adj_matrix)):
        train_adj_lists[node_index] = list(np.nonzero(dense_adj_matrix[node_index])[0])
    _, groups = metis.part_graph(train_adj_lists, 5, seed=0)
    test_idx = np.where(np.asarray(groups) == 4)[0]
    train_total_idx = np.setdiff1d(np.arange(n), test_idx)

    # use subsamples from all train nodes for actual training (privacy amplification)
    train_total_idx = np.random.permutation(train_total_idx)
    train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]

    # generate train subgraph
    train_labels = g.labels[train_idx]
    train_adj_matrix = dense_adj_matrix[train_idx, :][:, train_idx]
    num_edges = sum(sum(train_adj_matrix))
    np.fill_diagonal(train_adj_matrix, 1)
    train_adj_matrix = sp.csr_matrix(train_adj_matrix)

    train_attr_matrix = dense_attr_matrix[train_idx, :]
    train_attr_matrix = sp.csr_matrix(train_attr_matrix)
    train_index = np.arange(len(train_idx))

    # generate test subgraph
    test_labels = g.labels[test_idx]
    test_adj_matrix = dense_adj_matrix[test_idx, :][:, test_idx]
    np.fill_diagonal(test_adj_matrix, 1)
    test_adj_matrix = sp.csr_matrix(test_adj_matrix)

    test_attr_matrix = dense_attr_matrix[test_idx, :]
    test_attr_matrix = sp.csr_matrix(test_attr_matrix)
    if sp.issparse(test_attr_matrix):
        test_attr_matrix = SparseRowIndexer(test_attr_matrix)
    test_index = np.arange(len(test_idx))

    return train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
           test_attr_matrix, test_index, n, class_number, d, num_edges

def compute_epsilon(steps, sigma, delta, sampling_rate):
    """Computes epsilon value for given hyper-parameters."""
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    rdp = compute_rdp(q=sampling_rate,
                      noise_multiplier=sigma,
                      steps=steps,
                      orders=orders)
    return get_privacy_spent(orders, rdp, target_delta=delta)[0]


def EM_Gumbel_Optimal(EM_eps, topk, svt_sensitivity, report_noise_val_eps, p_vector):
    p_vector_copy = np.copy(p_vector)
    p_vector_copy_2 = np.copy(p_vector)
    # clip
    for idx in range(len(p_vector)):
        ppr_value_ = p_vector_copy[idx]
        ppr_value_2 = p_vector_copy_2[idx]
        if ppr_value_ > svt_sensitivity:
            p_vector_copy[idx] = svt_sensitivity
        if ppr_value_2 > svt_sensitivity:
            p_vector_copy_2[idx] = svt_sensitivity

    gumbel_noise = np.random.gumbel(scale=2 * svt_sensitivity / EM_eps, size=p_vector.shape)
    p_vector_copy += gumbel_noise
    j_em = np.argsort(p_vector_copy)[-topk:]

    val_em = []
    for j in j_em:
        if report_noise_val_eps != 0:
            val = p_vector_copy_2[j] + np.random.laplace(scale=topk * svt_sensitivity / report_noise_val_eps)
        else:
            val = 1.0 / topk
        val_em.append(val)
    return list(j_em), val_em

