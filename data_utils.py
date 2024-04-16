import os.path

import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Any, Optional, Dict, Iterable, Union


@dataclass
class Location:
    term: str
    document: str
    begin_time: float
    end_time: float
    score: float = 1


def read_rttm(rttm_file: str) -> Tuple[dict, set, dict, dict]:
    'LEXEME 16257_A_20120709_025101_001451 1 0.13 0.44 alo <NA> <NA> <NA>'
    'islex utterance channel begin duration word e1 e2 e3'
    names = 'islex utterance channel begin duration word e1 e2 e3'.split()
    rttm = pd.read_csv(rttm_file, sep=' ', header=None,
                       names=names, na_filter=False)
    rttm = rttm[rttm['islex'] == 'LEXEME']
    rttm["end"] = rttm["begin"] + rttm["duration"]
    locations = rttm[['utterance', 'begin', 'end', 'word']]
    return locations


def read_ctm(ctm_file: str) -> pd.DataFrame:
    names = 'utterance channel begin duration word'.split()
    ctm = pd.read_csv(ctm_file, sep='\s+', header=None, names=names,
                      dtype={'utterance': str,
                             'channel': int,
                             'begin': float,
                             'duration': float,
                             'word': str,
                             }, na_filter=False)
    ctm = ctm[ctm['word'] != '<UNK>']
    ctm["end"] = ctm["begin"] + ctm["duration"]
    locations = ctm[['utterance', 'begin', 'end', 'word']]
    return locations

def read_txt(txt_file: str, has_uttid=True) -> pd.DataFrame:
    names = 'utterance text'.split() if has_uttid else ['text']
    txt = pd.read_csv(txt_file, sep='\t', header=None, names=names,
                      dtype={'utterance': str,
                             'text': str,
                             }, na_filter=False)
    return txt


def save_mats(dirname, filename='data.npy',
              mats=None, offsets=None, keys=None):
    if mats is not None:
        np.save(os.path.join(dirname, filename), mats)
    if offsets is not None:
        np.save(os.path.join(dirname, 'offsets_data.npy'), offsets)
    if keys is not None:
        with open(os.path.join(dirname, 'keys_data.txt'), 'w') as _key_out:
            for key in keys:
                _key_out.write(key.strip() + '\n')

class UtteranceIterator:
    def __init__(self, path: str, kind: str='default'):
        self.kind = kind
        self.path = path
        if self.kind == 'mmap':
            self.data = np.load(os.path.join(path, 'data.npy'), mmap_mode='r')
        else:
            self.data = np.load(os.path.join(path, 'data.npy'))
        self.offsets = np.load(os.path.join(path, 'offsets_data.npy'))
        self.utt_list = [_.strip() for _ in open(os.path.join(path, 'keys_data.txt'))]
        self.utt_dict = {x: i for i, x in enumerate(self.utt_list)}
        # Every self.referesh-th utterance will trigger a reload in mmap mode to limit memory growth.
        # Set to negative value to allow memory to grow until entire dataset is in memory
        self.refresh = 1000
        self.count = 0

    @property
    def shape(self):
        return self[0].shape

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, item):
        utterance = self.utt_list[item]
        return self.get_mat(utterance)

    def get_mat(self, utterance):
        uttid = self.utt_dict[utterance]
        if uttid == 0:
            beg = 0
        else:
            beg = self.offsets[uttid - 1]
        if uttid == len(self.offsets):
            end = len(self.data)
        else:
            end = self.offsets[uttid]

        self.count += 1
        if self.kind == 'mmap' and self.refresh > 0 and self.count % self.refresh == 0:  # Hack to limit memory growth
            self.data = np.load(os.path.join(self.path, 'data.npy'), mmap_mode='r')

        return torch.from_numpy(np.copy(self.data[beg:end]))

def load_data(datadir: str, dataloader_kind='default'):
    if dataloader_kind == 'mmap':
        data = UtteranceIterator(datadir)
        keys = data.utt_dict
    else:
        offsets = np.load(os.path.join(datadir, 'offsets_data.npy'))
        data = []
        with open(os.path.join(datadir, 'data.npy'), 'rb') as _f:
            while True:
                try:
                    data.append(np.load(_f).astype('float32'))
                except ValueError:
                    data = np.concatenate(data)
                    break
        keys = {line.strip(): i for i, line
                in enumerate(open(os.path.join(datadir, 'keys_data.txt')))}
        data = [torch.from_numpy(x) for x in np.array_split(data, offsets)]
    return datadir, data, keys