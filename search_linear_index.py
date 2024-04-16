import os
import argparse
from typing import Dict

import numpy as np
import torch
import utils
import models
import data_utils


def to_lower(string_x: str, special: Dict[str, str] = None) -> str:
    if special is None:
        special = {}
    y = [x.lower() if x not in special.keys() else special[x]
         for x in string_x]
    y = [x for x in y if x.isalnum()]
    return ''.join(y).strip()


def load_from_dir(trainer_dir, device):
    model_classname = open(os.path.join(trainer_dir, 'nnet_kind.txt')).read().strip()
    model_kind = models.get_model(model_classname, searcher=True)
    model = model_kind.load_from_dir(trainer_dir, map_location=device)
    print(type(model))
    model.to(device)
    return model


def keywords2mat(keywords, tokenizer):
    keyword_mats = []
    for k in keywords:
        x = [tokenizer[a] for a in k if a in tokenizer]
        keyword_mats.append(torch.tensor(x))
    return keyword_mats


def compute_query_encoding(queries, model, device, batch_size=-1):
    model.to(device)
    model.eval()

    if batch_size < 0:
        batch_size = len(queries)
    query_split = [queries[i * batch_size: (i + 1) * batch_size] for i in
                    range(int(len(queries) // batch_size))]
    remainder = len(queries) % batch_size
    if remainder:
        query_split.append(queries[len(queries) - remainder:])
    all_query_encodings = []
    all_query_encodings = []
    with torch.set_grad_enabled(False):
        for query in query_split:
            lens = [len(q) for q in query]
            max_len = max(lens)
            # Extra dimension is used for padding
            query_batch = (model.query_encoder.input_dim - 1) * torch.ones(len(lens),
                                                                            max_len, dtype=int)
            query_mask = torch.zeros(len(lens), max_len).to(device)
            for i, q in enumerate(query):
                query_batch[i, :lens[i]] = q
                query_mask[i, :lens[i]] = 1
            query_batch = query_batch.to(device)
            query_dict = {'features': query_batch, 'lengths': torch.tensor(lens)}
            query_dict = model.encode_query(query_dict)
            query_enc = query_dict['features'].detach().cpu()
            all_query_encodings.append(query_enc)
    all_query_encodings = torch.cat(all_query_encodings, dim=0)
    return all_query_encodings


def search(model, query_encodings, document_index, threshold=0.5):
    # Everything here is done on cpu, since inner products are fast on cpu
    # and can be memory intensive depending on matrix sizes
    with torch.set_grad_enabled(False):
        query_encodings = query_encodings.to('cpu')
        document_index = torch.from_numpy(document_index.data)
        model.to('cpu')

        # Frame-wise inner products
        all_search_results = torch.einsum('qd, nd->qn', query_encodings, document_index).sigmoid()
    return all_search_results * (all_search_results >= threshold)


def detect_nonzeros(vec):
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    edges, = np.nonzero(np.diff((vec == 0) * 1))
    edge_vec = [edges + 1]
    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def split_hyps_self(hypotheses, doc_offsets, results):
    # Assumes hypotheses are sorted by beginning time
    oi = 0
    hi = 0

    beg = 0
    end = doc_offsets[0]

    return_hypotheses = []

    while hi < len(hypotheses):
        hb, he = hypotheses[hi][0], hypotheses[hi][1]
        if hb >= end:
            oi += 1
        elif beg >= he:
            hi += 1
        else:
            if he <= end:
                if results is not None:
                    score = np.median(results[hb:he])
                else:
                    score = 0
                return_hypotheses.append((oi, hb - beg, he - beg, score))
                hi += 1
            else:
                if results is not None:
                    score = np.median(results[hb:end])
                else:
                    score = 0
                return_hypotheses.append((oi, hb - beg, end - beg, score))
                hypotheses[hi][0] = end

        if oi > 0:
            beg = doc_offsets[oi - 1]
        if oi < len(doc_offsets) - 1:
            end = doc_offsets[oi]
        else:
            end = np.inf
    return return_hypotheses

def write_search_output(outdir, keyword_ids, keyword_list, search_results, document_index,
                        min_scale=0.5,
                        ):
    utils.chk_mkdir(os.path.join(outdir, 'results'))
    docs = list(document_index.utt_list)

    for term_id, term, results in zip(keyword_ids, keyword_list, search_results):
        with open(os.path.join(outdir, 'results', term_id), 'w', encoding='utf-8') as _res:
            hypotheses = [list(_) for _ in detect_nonzeros(results)]
            hypotheses = split_hyps_self(hypotheses, document_index.offsets, results)
            for hyp in hypotheses:
                utterance, f_beg, f_end, score = hyp
                duration = f_end - f_beg
                if duration < len(''.join(term.split())) * min_scale:
                    continue
                utterance = docs[utterance]
                towrite = [term_id, utterance, f_beg, f_end, score
                ]
                towrite = [str(_) for _ in towrite]
                _res.write(' '.join(towrite) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        help='if specified, turns off gpu usage.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='pruning threshold')
    parser.add_argument('--min-time-scale', '--mn', type=float, default=1,
                        help='minimum allowed relative length of hypotheses')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='batch size for NN computations')
    parser.add_argument('--char-map', '--lower',
                        help='colon separated file that contains extra mappings between capital and small letters, '
                             'e.g. "İ": "i"`newline`"I": "ı" in Turkish')
    parser.add_argument('keywords_file',
                        help='tab separated list of keyword ids and the '
                             'corresponding keywords')
    parser.add_argument('index_dir',
                        help='directory that contains the document vector index')
    parser.add_argument('tokenizer',
                        help='pickle file containing the training token dictionary mapping letter to int')
    parser.add_argument('nnet_dir',
                        help='directory that contains the saved network files')
    parser.add_argument('outdir',
                        help='directory into which to store the result files')
    args = parser.parse_args()

    if args.no_cuda:
        device = 'cpu'
    else:
        device = 'cuda:0'

    document_index = data_utils.UtteranceIterator(args.index_dir, kind='mmap')
    model = load_from_dir(args.nnet_dir, device=device)
    tokenizer = utils.pkl_load(args.tokenizer)

    if args.char_map and os.path.isfile(args.char_map):
        special = {line.strip().split(':')[0]: line.strip().split(':')[1]
                    for line in open(args.char_map, encoding='UTF-8')}
    else:
        special = {}

    keyword_ids = [line.strip().split()[0] for line in open(args.keywords_file, encoding='utf-8').readlines()]
    keyword_list = [to_lower(''.join(line.strip().split()[1:]), special=special)
                    for line in open(args.keywords_file, encoding='utf-8').readlines()]
    queries = keywords2mat(keyword_list, tokenizer)

    batch_size = args.batch_size
    if batch_size < 0:
        batch_size = len(queries)
    query_split = [queries[i * batch_size: (i + 1) * batch_size] for i in
                    range(int(len(queries) // batch_size))]
    remainder = len(queries) % batch_size
    if remainder:
        query_split.append(queries[len(queries) - remainder:])
    model.eval()
    all_query_encodings = compute_query_encoding(queries, model, device, args.batch_size)

    search_output = search(model, all_query_encodings, document_index, threshold=args.threshold)
    write_search_output(
        args.outdir,
        keyword_ids,
        keyword_list,
        search_output,
        document_index,
        min_scale=args.min_time_scale,
        )


if __name__ == '__main__':
    main()
