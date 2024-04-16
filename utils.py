import os
import pickle
import time

import numpy as np

class SimpleLoader:
    def initialize_args(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class StopWatch:
    def __init__(self, keys=('main',)):
        self.begin_times = {key: 0. for key in keys}
        self.end_times = {key: 0. for key in keys}
        self.elapsed_times = {key: 0. for key in keys}

    def add_watch(self, keys: tuple):
        for key in keys:
            if key not in self.begin_times.keys():
                self.begin_times[key] = 0.
                self.end_times[key] = 0.
                self.elapsed_times[key] = 0.

    def start(self, key='main'):
        self.begin_times[key] = time.time()

    def stop(self, key='main'):
        self.end_times[key] = time.time()

    def track(self, key='main'):
        self.elapsed_times[key] += (self.end_times[key] - self.begin_times[key])

    def reset_tracker(self, key='main'):
        self.elapsed_times[key] = 0.

    def run(self, key, func, *args, **kwargs):
        self.start(key)
        return_value = func(*args, **kwargs)
        self.stop(key)
        self.track(key)
        return return_value

    def __repr__(self):
        return self.elapsed_times.__repr__()


def since(t0):
    return time.time() - t0


def reduce_lr(history, lr, cooldown=0, patience=5, mode='min',
              difference=0.001, lr_scale=0.5, lr_min=0.00001,
              cool_down_patience=None):
    if cool_down_patience and cooldown <= cool_down_patience:
        return lr, cooldown+1
    assert lr_scale < 1
    if mode == 'max':
        h = [-a for a in history]
    else:
        h = history
    history = h
    len_hist = len(history)
    if len_hist <= patience:
        return lr, cooldown + 1
    recent_history = history[len_hist-patience:len_hist]
    antiquity = history[0:len_hist-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return lr, cooldown+1
    else:
        return max(lr*lr_scale, lr_min), 0


def stop_early(history, patience=5, mode='min', difference=0.001):
    if mode == 'max':
        h = [-a for a in history]
    else:
        h = history
    history = h
    len_hist = len(history)
    if len_hist <= patience:
        return False
    recent_history = history[len_hist-patience:len_hist]
    antiquity = history[0:len_hist-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return False
    else:
        return True


def chk_mkdir(dirname):
    if isinstance(dirname, str):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
    else:
        try:
            dirnames = iter(dirname)
            for d in dirnames:
                chk_mkdir(d)
        except TypeError:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)


def pkl_save(obj, filename):
    outdir = os.path.dirname(filename)
    chk_mkdir(outdir)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
