import numpy as np
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

def sim_kernel(dot_product):
    return dot_product.power(3).maximum(0)


def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal())
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn


def minimum_affinity_matrix(G):
    G = G.minimum(G.transpose())
    G.eliminate_zeros()
    return G


def find_trunc_graph(qs, W, levels = 3):
    right = np.nonzero(qs > 0)[0]
    left = list(W.nonzero()[1])
    needed_idxs = np.concatenate((right, left))
    needed_idxs = list(set(needed_idxs))
    w_idxs, W_trunk = np.array(needed_idxs), W[needed_idxs,:][:,needed_idxs]
    return w_idxs, W_trunk


def dfs_trunk(qsims, W, alpha=0.99,  maxiter=8, tol=1e-3, top_k=None):
    ranks = []
    for i in tqdm(range(qsims.shape[0]), total=qsims.shape[0], ascii=True):
        qs = qsims[i, :].transpose()
        _, cur_rank = task_dfs_loop(i, qs, alpha, W, tol, maxiter, top_k)
        ranks.append(cur_rank)

    ranks = np.concatenate(ranks, axis=-1)
    return ranks


def dfs_trunk_multithread(qsims, W, alpha=0.99,  maxiter=8, tol=1e-3, top_k=None, worker=4):
    with ThreadPoolExecutor(max_workers=worker) as executor:
        ranks_futures = []
        n = qsims.shape[0]
        for i in range(n):
            qs = qsims[i, :].transpose()
            ranks_futures.append(executor.submit(task_dfs_loop, i, qs, alpha, W, tol, maxiter, top_k))

        ranks_dict = {}
        for f in tqdm(as_completed(ranks_futures), total=len(ranks_futures), ascii=True):
            i, cur_rank = f.result()
            ranks_dict[i] = cur_rank
        ranks = [ranks_dict[i] for i in range(len(ranks_dict))]

    ranks = np.concatenate(ranks, axis=-1)
    return ranks


def dfs_trunk_multiprocess(qsims, W, alpha=0.99,  maxiter=8, tol=1e-3, top_k=None, worker=4):
    with ProcessPoolExecutor(max_workers=worker) as executor:
        ranks_futures = []
        n = qsims.shape[0]
        for i in range(n):
            qs = qsims[i, :].transpose()
            ranks_futures.append(executor.submit(task_dfs_loop, i, qs, alpha, W, tol, maxiter, top_k))

        ranks_dict = {}
        for f in tqdm(as_completed(ranks_futures), total=len(ranks_futures), ascii=True):
            i, cur_rank = f.result()
            ranks_dict[i] = cur_rank
        ranks = [ranks_dict[i] for i in range(len(ranks_dict))]

    ranks = np.concatenate(ranks, axis=-1)
    return ranks


def task_dfs_loop(i, qs, alpha, W, tol, maxiter, top_k):
    w_idxs, W_trunk = find_trunc_graph(qs, W, 2)
    Wn = normalize_connection_graph(W_trunk)
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    f, inf = s_linalg.minres(Wnn, qs[w_idxs].toarray(), tol=tol, maxiter=maxiter)
    ranks = w_idxs[np.argsort(-f.reshape(-1))]
    missing = np.setdiff1d(np.arange(W.shape[1]), ranks)
    cur_rank = np.concatenate([ranks.reshape(-1,1), missing.reshape(-1,1)], axis = 0)
    if top_k is not None:
        cur_rank = cur_rank[:top_k]
    return i, cur_rank


def fsr_rankR(qsims, Wn, alpha=0.99, R=2000, top_k=None):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 = vc.dot(p2)
    vcT = vc.transpose()
    qsimsT = qsims.transpose()
    ranks = []
    # for i in tqdm(range(qsims.shape[0]), total=qsims.shape[0], ascii=True):
    for i in range(qsims.shape[0]):
        qsims_sparse = qsimsT.getcol(i)
        _, cur_rank = task_fsr_loop(i, qsims_sparse, vcT, p3, top_k)
        ranks.append(cur_rank)

    ranks = np.stack(ranks, axis=-1)
    return ranks


def fsr_rankR_multithread(qsims, Wn, alpha=0.99, R=2000, top_k=None, worker=4):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 = vc.dot(p2)
    vcT = vc.transpose()
    qsimsT = qsims.transpose()
    with ThreadPoolExecutor(max_workers=worker) as executor:
        ranks_futures = []
        n = qsims.shape[0]

        for i in range(n):
            qsims_sparse = qsimsT.getcol(i)
            ranks_futures.append(executor.submit(task_fsr_loop, i, qsims_sparse, vcT, p3, top_k))

        ranks_dict = {}
        for f in tqdm(as_completed(ranks_futures), total=len(ranks_futures), ascii=True):
            i, cur_rank = f.result()
            ranks_dict[i] = cur_rank
        ranks = [ranks_dict[i] for i in range(len(ranks_dict))]

    ranks = np.stack(ranks, axis=-1)
    return ranks


def fsr_rankR_multiprocess(qsims, Wn, alpha=0.99, R=2000, top_k=None, worker=4):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 = vc.dot(p2)
    vcT = vc.transpose()
    qsimsT = qsims.transpose()
    with ProcessPoolExecutor(max_workers=worker) as executor:
        ranks_futures = []
        n = qsims.shape[0]

        for i in range(n):
            qsims_sparse = qsimsT.getcol(i)
            ranks_futures.append(executor.submit(task_fsr_loop, i, qsims_sparse, vcT, p3, top_k))

        ranks_dict = {}
        for f in tqdm(as_completed(ranks_futures), total=len(ranks_futures), ascii=True):
            i, cur_rank = f.result()
            ranks_dict[i] = cur_rank
        ranks = [ranks_dict[i] for i in range(len(ranks_dict))]

    ranks = np.stack(ranks, axis=-1)
    return ranks


def task_fsr_loop(i, qsims_sparse, vcT, p3, top_k):
    p1 = vcT.dot(qsims_sparse)
    diff_sim = p3 * p1
    diff_sim = diff_sim.toarray().reshape(-1,)
    cur_rank = np.argsort(-diff_sim, axis=0)
    if top_k is not None:
        cur_rank = cur_rank[:top_k]
    return i, cur_rank

