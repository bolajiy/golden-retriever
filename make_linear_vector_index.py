import os
import argparse
from typing import Dict
import utils
import numpy as np
import torch
import models
import data_utils

def to_lower(string_x: str, tr=False, special: Dict = None) -> str:
    if special is None:
        special = {}
    y = [x.lower() if x not in special.keys() else special[x]
         for x in string_x]
    y = [x for x in y if x.isalnum()]
    return ''.join(y).strip()


class LinearVectorIndex(utils.SimpleLoader):
    def __init__(self, model_dir, docs_datadir, device, outdir, **kwargs):
        self.params_dict = {'batch_size': 64,}
        super().initialize_args(kwargs)
        self.model_dir = model_dir
        self.device = device
        self.docs_datadir = docs_datadir
        self.doc_keys = {line.strip(): i for i, line
                         in enumerate(open(os.path.join(docs_datadir, 'keys_data.txt')))}
        self.doc_offsets = np.load(os.path.join(docs_datadir, 'offsets_data.npy'))
        data = []
        with open(os.path.join(docs_datadir, 'data.npy'), 'rb') as _f:
            while True:
                try:
                    data.append(np.load(_f).astype('float32'))
                except ValueError:
                    self.doc_data = np.concatenate(data)
                    break
        self.doc_data = [torch.from_numpy(x) for x in np.array_split(self.doc_data, self.doc_offsets)]
        self.outdir = outdir
        utils.chk_mkdir(self.outdir)
        self.cache = dict()
        self.model = self.load_from_dir(model_dir)

    def load_from_dir(self, trainer_dir, model_kind=models.CompositeModel):
        if os.path.isfile(os.path.join(trainer_dir, 'nnet_kind.txt')):
            model_classname = open(os.path.join(trainer_dir, 'nnet_kind.txt')).read().strip()
            model_kind = models.get_model(model_classname, searcher=True)
        model = model_kind.load_from_dir(trainer_dir, map_location=self.device)
        print(type(model))
        model.to(self.device)
        return model

    def index_documents(self, documents=None):
        if documents is None:
            documents = self.doc_data

        if not os.path.isfile(os.path.join(self.outdir, 'data.npy')):
            batch_size = self.batch_size
            document_split = [documents[i * batch_size: (i + 1) * batch_size] for i in
                              range(int(len(documents) // batch_size))]
            remainder = len(documents) % batch_size
            if remainder:
                document_split.append(documents[len(documents) - remainder:])
            self.model.eval()
            all_indices = []
            new_lengths = []
            with torch.set_grad_enabled(False):
                for document in document_split:
                    lens = [len(d) for d in document]
                    max_len = max(lens)
                    document_batch = torch.zeros(len(lens), max_len, document[0].size(-1))
                    for i, d in enumerate(document):
                        document_batch[i, :lens[i]] = d
                    document_batch = document_batch.to(self.device)
                    lengths = torch.tensor(lens)
                    document_dict = {'features': document_batch,
                                     'lengths': lengths}
                    document_dict = self.model.encode_document(document_dict)
                    for doc, doc_length in zip(document_dict['features'], document_dict['lengths']):
                        all_indices.append(doc[:doc_length].detach().cpu())
                        new_lengths.append(doc_length)
            document_index = torch.cat(all_indices, dim=0).numpy()
            self.doc_offsets = np.cumsum(new_lengths[:-1])

            data_utils.save_mats(
                self.outdir,
                filename='data.npy',
                mats=document_index,
                offsets=self.doc_offsets,
                keys=self.doc_keys,
                )
        else:
            index_file = os.path.join(self.outdir, 'data.npy')
            print(f"Index already exists at {index_file}, skipping")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        help='if specified, turns off gpu usage.')
    parser.add_argument('--search-kind', default='IndexableSearch',
                        help='kind of search as defined in search.py')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for NN computations')
    parser.add_argument('datadir',
                        help='directory that contains the document matrix')
    parser.add_argument('nnet_dir',
                        help='directory that contains the saved network files')
    parser.add_argument('outdir',
                        help='directory into which to store the resulting vector index')
    args = parser.parse_args()

    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda:0'

    vector_index = LinearVectorIndex(args.nnet_dir,
                                     args.datadir,
                                     torch.device(device),
                                     args.outdir,
                                     batch_size=args.batch_size,
                                     )
    vector_index.index_documents()


if __name__ == '__main__':
    main()
