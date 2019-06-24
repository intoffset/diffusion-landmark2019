#!/usr/bin/env python
# coding: utf-8

"""
Diffusionをかける

入出力は以下の想定

../input
|-- index_features.npy
|-- index_ids.pickle
|-- test_features.npy
`-- test_ids.pickle

../similarity
|-- query-index-similarity.npz
`-- index-index-similarity.npz

../output/
`-- submission.csv

"""

from absl import app, flags
import os

import numpy as np
from tqdm import tqdm

from landmark_diffusion.model import FeatureContainer, Submission, SparseSimilarity
from landmark_diffusion.diffussion_sparse import sim_kernel, minimum_affinity_matrix, normalize_connection_graph, \
    fsr_rankR, fsr_rankR_multiprocess, fsr_rankR_multithread
from landmark_diffusion.util import StopWatch

flags.DEFINE_string('input_dir', '../input', "input directory for diffusion", short_name='i')
flags.DEFINE_string('similarity_dir', '../similarity', "input directory of similarity", short_name='s')
flags.DEFINE_string('output_file', '../output/submissoin.csv', "output file for submission", short_name='o')
flags.DEFINE_integer('top_k', 100, "number of predicted samples per query")
flags.DEFINE_integer('R', 1000,
                     "diffusion parameter: number of eigenvalue. R < num_candidate if num_candidate is specified")
flags.DEFINE_float('alpha', 0.9, "diffusion parameter")
flags.DEFINE_bool('verbose', False, "whether to report time")

flags.DEFINE_enum('parallel', None, ['t', 'p', 'thread', 'process'], "parallel execusion for diffusion")
flags.DEFINE_integer('worker', 4, "parallel worker")

FLAGS = flags.FLAGS

from landmark_diffusion.constant import *


def main(argv=None):

    # Diffusionのパラメータ
    R = FLAGS.R
    alpha = FLAGS.alpha

    # Set file path
    path_test_feature = os.path.join(FLAGS.input_dir, FILENAME_TEST_FEATURE)
    path_test_index = os.path.join(FLAGS.input_dir, FILENAME_TEST_ID)
    path_index_feature = os.path.join(FLAGS.input_dir, FILENAME_INDEX_FEATURE)
    path_index_index = os.path.join(FLAGS.input_dir, FILENAME_INDEX_ID)

    path_test_index_similarity = os.path.join(FLAGS.similarity_dir, FILENAME_TEST_INDEX_SIM)
    path_index_index_similarity = os.path.join(FLAGS.similarity_dir, FILENAME_INDEX_INDEX_SIM)

    # デバッグ用のストップウォッチを起動
    stop_watch = StopWatch(verbose=FLAGS.verbose)
    stop_watch.start()

    # 特徴量ファイルを読み込み
    print("Loading feature files")
    test_feature_container = FeatureContainer.load(path_test_feature, path_test_index)
    index_feature_container = FeatureContainer.load(path_index_feature, path_index_index)
    stop_watch.lap(message="Load feature files")

    # # TODO デバッグ用に間引き
    # test_feature_container = test_feature_container.get_item(range(min(300, len(test_feature_container))))
    # index_feature_container = index_feature_container.get_item(range(min(3000, len(index_feature_container))))

    print("num samples in test is {}".format(test_feature_container.num_sample))
    print("num samples in index is {}".format(index_feature_container.num_sample))
    assert test_feature_container.num_feature == index_feature_container.num_feature
    print("num feature in index/test is {}".format(index_feature_container.num_feature))

    # 類似度ファイルを読み込み
    print("Load feature files")
    test_index_similarity = SparseSimilarity.load(path_test_index_similarity)
    index_index_similarity = SparseSimilarity.load(path_index_index_similarity)
    stop_watch.lap(message="Load similarity files")

    # Copied from https://github.com/ducha-aiki/manifold-diffusion/blob/master/example_evaluate_with_diff.py
    # Sparse行列を扱うために一部書き換え
    sim = test_index_similarity.sparse_sim
    qsim = sim_kernel(sim)

    # TODO: 元コードでもしてるけどsim_kernelを2重でかけるのは正しいのだろうか？
    qsim = sim_kernel(qsim)

    stop_watch.lap("Convert similarity in query and index")

    # Diffusion用の事前計算
    print("Calculate connection graph")
    A = index_index_similarity.sparse_sim
    stop_watch.lap(message="Calculate affinity matrix for index")

    W = sim_kernel(A).T
    # 予めTOP-Kをとっているので、転地行列との最小値を求めるだけ
    W = minimum_affinity_matrix(W)
    Wn = normalize_connection_graph(W)
    stop_watch.lap(message="Calculate connection graph")

    submission_diffusion = Submission.empty()

    if FLAGS.parallel is None:
        fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, R, FLAGS.top_k)
    elif FLAGS.parallel in ['p', 'process']:
        fast_spectral_ranks = fsr_rankR_multiprocess(qsim, Wn, alpha, R, FLAGS.top_k, worker=FLAGS.worker)
    elif FLAGS.parallel in ['t', 'thread']:
        fast_spectral_ranks = fsr_rankR_multithread(qsim, Wn, alpha, R, FLAGS.top_k, worker=FLAGS.worker)
    else:
        raise ValueError

    fast_spectral_ranks = np.asarray(fast_spectral_ranks[:FLAGS.top_k].T)

    stop_watch.lap("Calc fast spectral ranks")

    num_query = len(test_feature_container)
    for i in tqdm(range(num_query), total=num_query, ascii=True):
        query_id = test_feature_container.get_id(i)
        _cur_rank = fast_spectral_ranks[i]
        predicted_features = index_feature_container.get_item(_cur_rank)
        submission_diffusion.append(query_id, predicted_features.ids)

    stop_watch.lap("Append result to submission (in memory)")

    dir_out = os.path.dirname(FLAGS.output_file)
    os.makedirs(dir_out, exist_ok=True)

    submission_diffusion.save(FLAGS.output_file)

    stop_watch.lap("Save file as submission format")

    if FLAGS.verbose:
        stop_watch.report()


if __name__ == '__main__':
    app.run(main)
