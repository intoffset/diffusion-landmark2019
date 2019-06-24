#!/usr/bin/env python
# coding: utf-8

"""
Diffusionに必要な類似度行列を算出する

入出力は以下の想定

../input
|-- test_features.npy
|-- test_ids.pickle
|-- index_features.npy
`-- index_ids.pickle

../similarity
|-- submission.csv # this is input file, not submit target
|-- query-index-similarity.npz
`-- index-index-similarity.npz

../output/
`-- submission.csv

"""

from absl import app, flags
import os
import numpy as np

from landmark_diffusion.model import FeatureContainer, SparseSimilarity
from landmark_diffusion.util import StopWatch

flags.DEFINE_string('input_dir', '../input', "input directory for diffusion", short_name='i')
flags.DEFINE_string('output_dir', '../output', "output directory for similarity", short_name='o')
flags.DEFINE_integer('K', 100, "diffusion parameter: number of K-NN samples in index group")
flags.DEFINE_integer('QUERYKNN', 10, "diffusion parameter: number of K-NN samples in query and index")
flags.DEFINE_bool('norm', True, "whether apply normalize to sample")
flags.DEFINE_bool('gpu', True, "whether to use gpu for Faiss")
flags.DEFINE_bool('verbose', False, "whether to report time")

FLAGS = flags.FLAGS

from landmark_diffusion.constant import *


def main(argv=None):

    # Diffusionのパラメータ
    K = FLAGS.K
    QUERYKNN = FLAGS.QUERYKNN

    # Set file path
    path_test_feature = os.path.join(FLAGS.input_dir, FILENAME_TEST_FEATURE)
    path_test_index = os.path.join(FLAGS.input_dir, FILENAME_TEST_ID)
    path_index_feature = os.path.join(FLAGS.input_dir, FILENAME_INDEX_FEATURE)
    path_index_index = os.path.join(FLAGS.input_dir, FILENAME_INDEX_ID)

    # デバッグ用のストップウォッチを起動
    stop_watch = StopWatch(verbose=FLAGS.verbose)
    stop_watch.start()

    # 特徴量ファイルを読み込み
    print("Load feature files")
    test_feature_container = FeatureContainer.load(path_test_feature, path_test_index)
    index_feature_container = FeatureContainer.load(path_index_feature, path_index_index)
    stop_watch.lap(message="Load feature files")

    # 各サンプルに対してL2標準化
    if FLAGS.norm:
        test_feature_container.l2_normalize()
        index_feature_container.l2_normalize()

    # # TODO デバッグ用に間引き
    # test_feature_container = test_feature_container.get_item(range(300))
    # index_feature_container = index_feature_container.get_item(range(3000))

    # test(query)とindexの間の類似度を算出
    query_index_sim = SparseSimilarity.from_features(
        test_feature_container.features, index_feature_container.features, QUERYKNN, gpu=FLAGS.gpu)
    stop_watch.lap(message="Calculate similarity in query and index")

    index_index_sim = SparseSimilarity.from_features(
        index_feature_container.features, index_feature_container.features, K, gpu=FLAGS.gpu)
    stop_watch.lap(message="Calculate similarity in index group")

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    query_index_sim.save(os.path.join(FLAGS.output_dir, FILENAME_TEST_INDEX_SIM))
    stop_watch.lap(message="Save similarity in query and index")

    index_index_sim.save(os.path.join(FLAGS.output_dir, FILENAME_INDEX_INDEX_SIM))
    stop_watch.lap(message="Save similarity in index group")

    if FLAGS.verbose:
        stop_watch.report()


if __name__ == '__main__':
    app.run(main)
