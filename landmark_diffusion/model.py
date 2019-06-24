import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from landmark_diffusion import util


class FeatureContainer:
    """特徴量を扱うクラス"""

    def __init__(self, ids, features):
        assert len(ids) == features.shape[0]
        self.ids = ids
        self.features = features

    def __len__(self):
        return self.num_sample

    @property
    def num_sample(self):
        return self.features.shape[0]

    @property
    def num_feature(self):
        return self.features.shape[1]

    @classmethod
    def load(cls, path_features, path_indexes):
        features = np.load(path_features)
        with open(path_indexes, 'rb') as f:
            indexes = pickle.load(f)
        indexes = indexes.values
        return FeatureContainer(indexes, features)

    def get_id(self, index):
        return self.ids[index]

    def get_item(self, index):
        """indexのFeatureContainerを返す"""
        if isinstance(index, int):
            # indexがスカラーの場合
            return FeatureContainer(np.reshape(self.ids[index], (1,)), np.reshape(self.features[index], (1, -1)))
        else:
            # indexがリストorタプルの場合
            return FeatureContainer(self.ids[index], self.features[index])

    def get_subset(self, ids):
        """ idsにself.idsがマッチするFeatureContainerを返す
        NOTE: self.indexesに含まれないidがlist_indexesにあってもよい
        """
        ids = self.ids[[x in ids for x in self.ids]]
        features = self.features[[x in ids for x in self.ids], :]
        return FeatureContainer(ids, features)

    def id_to_row_dict(self):
        return {id: i for i, id in enumerate(self.ids)}

    def l2_normalize(self):
        """featuresを各サンプルごとに標準化する。inplace挙動"""
        l2 = np.linalg.norm(self.features, ord=2, axis=1, keepdims=True)
        l2[l2 == 0] = 1
        self.features /= l2

    @classmethod
    def similarity(cls, query_container, index_container, sim_fn='inner'):
        """類似度を返す関数。内積かcosine距離に対応"""

        dtype = query_container.features.dtype

        # TODO 計算効率化のためにFP16に変換
        if_fp16 = index_container.features.astype(np.float16)
        qf_fp16 = query_container.features.astype(np.float16)
        if sim_fn == 'inner':
            ret = util.sim_inner(if_fp16, qf_fp16)
        if sim_fn == 'cosine':
            ret = util.sim_cosine(if_fp16, qf_fp16)
        if sim_fn == 'inner_dask':
            ret = util.sim_inner_dask(if_fp16, qf_fp16)
        else:
            ValueError

        return ret.astype(dtype)

    @classmethod
    def similarity_inner_product_dask(self, query_container, index_container):
        """featureの類似度を検索。内積を取る
        :return 類似度matrix (num index, num query)
        """
        sim = util.sim_inner_dask(index_container.features.astype(np.float16), query_container.features.astype(np.float16))
        return sim

    @classmethod
    def similarity_inner_product(self, query_container, index_container):
        """featureの類似度を検索。内積を取る
        :return 類似度matrix (num index, num query)
        """
        sim = util.sim_inner(index_container.features.astype(np.float16), query_container.features.astype(np.float16))
        return sim

    @classmethod
    def similarity_cosine_distance(self, query_container, index_container):
        """featureの類似度を算出。cosine距離版
        :return 類似度matrix (num index, num query)
        """
        sim = util.sim_cosine(index_container.features.astype(np.float16), query_container.features.astype(np.float16))
        return sim


class Submission:
    """Submission形式のデータを扱うクラス"""

    def __init__(self, ids, images_ids):
        self.ids = ids
        self.images_ids = images_ids

    def __len__(self):
        return self.ids.shape[0]

    @classmethod
    def load(cls, path_submission, top_k=None):
        submission = pd.read_csv(path_submission)
        ids = submission['id'].values
        if top_k is None:
            images_ids = np.array(list(map(lambda x: x.split(' '), submission['images'].values)))
        else:
            # Submissionデータを間引く用
            def _map_fn(x):
                x = x.split(' ')
                k = min(len(x), top_k)
                x = x[:k]
                return x
            images_ids = np.array(list(map(_map_fn, submission['images'].values)))

        return Submission(ids, images_ids)

    @classmethod
    def empty(cls):
        ids = np.empty((0,), dtype=object)
        images_ids = np.empty((0,), dtype=object)
        return Submission(ids, images_ids)

    def append(self, id, image_ids):
        self.ids = np.append(self.ids, id)

        # TODO Tupleをappendしようとすると連結されてしまうのでしょうがなくこういう書き方。リファクタしたい
        self.images_ids = np.append(self.images_ids, None)
        self.images_ids[-1] = image_ids

    def get_by_id(self, id):
        images_ids = self.images_ids[np.where(self.ids == id)]
        return Submission(np.array([id]), images_ids)

    def get_id(self, i):
        return self.ids[i]

    def get_images_ids(self, i):
        return self.images_ids[i]

    def get_images_ids_union(self):
        return set().union(*self.images)

    def save(self, path_submission):
        """Kaggleのsubmission形式で出力する"""
        submission = pd.DataFrame({'id': self.ids,
                                   'images': list(map(lambda x: ' '.join(x), self.images_ids))})
        submission.to_csv(path_submission, index=False)


class SparseSimilarity:

    def __init__(self, sparse_sim):
        self.sparse_sim = sparse_sim

    @classmethod
    def from_features(self, query_features, index_features, KNN, gpu=False):
        if not gpu:
            D, I, (Nq, Nd) = _sim_faiss(query_features, index_features, KNN)
        else:
            D, I, (Nq, Nd) = _sim_faiss_gpu(query_features, index_features, KNN)

        rows = np.tile(np.reshape(np.arange(Nq), [Nq, -1]), KNN).flatten()
        cols = I.flatten()
        values = D.flatten()
        sparse_sim = csr_matrix((values, (rows, cols)), shape=(Nq, Nd))

        return SparseSimilarity(sparse_sim)


    @classmethod
    def load(self, path_file):
        return SparseSimilarity(load_npz(path_file))

    def save(self, path_file):
        save_npz(path_file, self.sparse_sim)


def _sim_faiss(query_features, index_features, KNN):
    """Faissでsimilarity"""
    import faiss
    assert query_features.shape[1] == index_features.shape[1]
    dim = query_features.shape[1]
    Nq = query_features.shape[0]
    Nd = index_features.shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(index_features)
    D, I = index.search(query_features, KNN)
    return D, I, (Nq, Nd)


def _sim_faiss_gpu(query_features, index_features, KNN):
    """Faissでsimilarity with GPU"""
    import faiss
    assert query_features.shape[1] == index_features.shape[1]
    dim = query_features.shape[1]
    Nq = query_features.shape[0]
    Nd = index_features.shape[0]

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(index_features)
    D, I = index.search(query_features, KNN)
    return D, I, (Nq, Nd)

