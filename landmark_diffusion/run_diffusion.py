#!/usr/bin/env python
# coding: utf-8

"""
Diffusionをかける

入出力は以下の想定

../input
|-- index_features.npy
|-- index_ids.pickle
|-- submission.csv
|-- test_features.npy
`-- test_ids.pickle

../output/
`-- submission.csv

"""

from absl import app, flags
import os

import numpy as np
from tqdm import tqdm

from landmark_diffusion.model import FeatureContainer, Submission
from landmark_diffusion.diffussion import sim_kernel, topK_W, normalize_connection_graph, fsr_rankR
from landmark_diffusion.util import StopWatch, sim_cosine, sim_inner

flags.DEFINE_string('input_dir', '../input', "input directory for diffusion", short_name='i')
flags.DEFINE_string('output_file', '../output/submissoin.csv', "output file for submission", short_name='o')
flags.DEFINE_integer('num_candidate', None, "number of candidate for diffusion process")
flags.DEFINE_integer('top_k', 100, "number of predicted samples per query")
flags.DEFINE_integer('R', 1000,
                     "diffusion parameter: number of eigenvalue. R < num_candidate if num_candidate is specified")
flags.DEFINE_integer('K', 100, "diffusion parameter: number of K-NN samples in index group")
flags.DEFINE_integer('QUERYKNN', 10, "diffusion parameter: number of K-NN samples in query and index")
flags.DEFINE_float('alpha', 0.9, "diffusion parameter")
flags.DEFINE_enum('sim_fn', 'inner', ['inner', 'inner_dask', 'cosine'], "similarity function")
flags.DEFINE_bool('verbose', False, "whether to report time")

FLAGS = flags.FLAGS

FILENAME_TEST_FEATURE = 'test_features.npy'
FILENAME_TEST_INDEX = 'test_ids.pickle'
FILENAME_INDEX_FEATURE = 'pca_index_features.npy'
FILENAME_INDEX_INDEX = 'index_ids.pickle'


def main(argv=None):

    # Diffusionのパラメータ
    R = FLAGS.R
    K = FLAGS.K
    QUERYKNN = FLAGS.QUERYKNN
    alpha = FLAGS.alpha

    # Set file path
    path_test_feature = os.path.join(FLAGS.input_dir, FILENAME_TEST_FEATURE)
    path_test_index = os.path.join(FLAGS.input_dir, FILENAME_TEST_INDEX)
    path_index_feature = os.path.join(FLAGS.input_dir, FILENAME_INDEX_FEATURE)
    path_index_index = os.path.join(FLAGS.input_dir, FILENAME_INDEX_INDEX)

    # デバッグ用のストップウォッチを起動
    stop_watch = StopWatch(verbose=FLAGS.verbose)
    stop_watch.start()

    # 特徴量ファイルを読み込み
    print("Load feature files")
    test_feature_container = FeatureContainer.load(path_test_feature, path_test_index)
    index_feature_container = FeatureContainer.load(path_index_feature, path_index_index)
    stop_watch.lap(message="Load feature files")

    # # TODO デバッグ用に間引き
    # test_feature_container = test_feature_container.get_item(range(min(300, len(test_feature_container))))
    # index_feature_container = index_feature_container.get_item(range(min(3000, len(index_feature_container))))

    # test(query)とindexの間の類似度を算出
    sim = FeatureContainer.similarity(test_feature_container, index_feature_container, sim_fn=FLAGS.sim_fn)

    stop_watch.lap(message="Calculate similarity in query and index")

    # Copied from https://github.com/ducha-aiki/manifold-diffusion/blob/master/example_evaluate_with_diff.py
    qsim = sim_kernel(sim).T

    sortidxs = np.argsort(-qsim, axis = 1)

    for i in range(len(qsim)):
        qsim[i, sortidxs[i,QUERYKNN:]] = 0

    # TODO: 元コードでもしてるけどsim_kernelを2重でかけるのは正しいのだろうか？
    qsim = sim_kernel(qsim)

    stop_watch.lap("Convert similarity in query and index")

    # Diffusion用の事前計算
    print("Calculate connection graph")
    A = FeatureContainer.similarity(index_feature_container, index_feature_container, sim_fn=FLAGS.sim_fn)
    stop_watch.lap(message="Calculate affinity matrix for index")

    W = sim_kernel(A).T
    W = topK_W(W, K)
    Wn = normalize_connection_graph(W)
    stop_watch.lap(message="Calculate connection graph")

    submission_diffusion = Submission.empty()

    fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, R)
    fast_spectral_ranks = np.asarray(fast_spectral_ranks[:FLAGS.top_k].T)

    stop_watch.lap("Calc fast spectral ranks")

    num_query = len(test_feature_container)
    for i in range(num_query):
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
