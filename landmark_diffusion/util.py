from time import time

import dask.array as da
import numpy as np
import scipy.spatial as sp


def sim_inner_dask(X, Q, verbose=False):
    # 内積
    stop_watch = StopWatch(verbose=verbose)
    stop_watch.start()
    QT = Q.T
    stop_watch.lap("Transpose matrix")
    dadot = da.dot(X, QT)
    stop_watch.lap("Define dot function by da.dot")
    res = dadot.compute()
    stop_watch.lap("Compute inner product")
    return res


def sim_inner(X, Q):
    # 内積
    return np.dot(X, Q.T)


def sim_cosine(X, Q):
    # 1.0 - cosine距離
    # https://stackoverflow.com/a/30152675
    return 1. - sp.distance.cdist(X, Q, 'cosine')


class StopWatch:
    def __init__(self, verbose):
        self.reset()
        self.verbose = verbose

    def reset(self):
        self.start_time = None
        self.last_time = None

        # Lap用のメンバ
        self.index_lap = 0
        self.laps = []
        self.messages = []

    def start(self):
        self.start_time = time()
        self.last_time = self.start_time

    def _print_lap(self, index_lap, lap, message):
        print("Lap {} {}[sec] : {} ".format(index_lap, lap, message))

    def lap(self, message=None):
        _last_time = time()

        self.index_lap += 1
        self.messages.append(message)
        self.laps.append(_last_time - self.last_time)
        self.last_time = _last_time
        if self.verbose:
            self._print_lap(self.index_lap, self.laps[-1], self.messages[-1])

    def report(self):
        print("Report from stop-watch:")
        for index_lap in range(self.index_lap):
            self._print_lap(index_lap + 1, self.laps[index_lap], self.messages[index_lap])




