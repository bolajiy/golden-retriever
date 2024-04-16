import os
import torch
from torch import nn
import json
from layers import CompositeModel, MultitaskCompositeModel


class SimpleLoader:
    def initialize_args(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class ModuleWithSaveLoad(nn.Module, SimpleLoader):
    def __init__(self, kwargs):
        if 'jsonfile' in kwargs.keys():
            jsonfile = kwargs['jsonfile']
            with open(jsonfile) as _json:
                self.params_dict = json.load(_json)
        else:
            self.params_dict = kwargs
        self.model_name = 'ModuleWithSaveLoad'
        super(ModuleWithSaveLoad, self).initialize_args(self.params_dict)

    def save(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            json.dump(self.params_dict, _json)
        if hasattr(self, 'model_name'):
            with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
                _kind.write(self.model_name)
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None):
        net = cls(os.path.join(nnetdir, 'nnet.json'))
        state_dict = torch.load(os.path.join(nnetdir, 'nnet.mdl'),
                                map_location=map_location)
        net.to(map_location)
        net.load_state_dict(state_dict)
        return net


class SimpleSearchNet(nn.Module):
    '''Dual-encoder model with query encoder and document encoder'''
    def __init__(self, query_encoder, document_encoder, query_sum=False, summarizer='sum', **kwargs):
        super().__init__()
        self.document_encoder = document_encoder
        self.query_encoder = query_encoder
        self.query_sum = query_sum
        self.summarizer = summarizer
        self.things_to_save = ('query_encoder', 'document_encoder')
        self.model_name = 'SimpleSearchNet'

    def forward(self, query, document, return_encoding=False, exhaustive=False, **kwargs):
        query_enc, document_enc = self.encode(query, document, **kwargs)
        search_result = self.search(query_enc, document_enc, exhaustive=exhaustive)
        if return_encoding:
            return search_result, query_enc, document_enc
        else:
            return search_result

    def summarize(self, input_sequence, mask=None, dim=1, **kwargs):
        x = self(input_sequence, **kwargs)
        if mask is not None:
            if mask.ndim < x.ndim:
                mask = mask.unsqueeze(-1).expand(*x.shape)
            x = x * mask
        return self.output_layer(x.sum(dim=dim))

    def encode_query(self, query):
        query_enc_dict = self.query_encoder(query)
        query_enc, query_lengths = query_enc_dict['features'], query_enc_dict['lengths']
        if not self.query_sum:
            # Return last timestep of encoder output
            query_inds = query_lengths - 1
            query_enc = query_enc[torch.arange(query_enc.size(0)), query_inds]
        else:
            # Return sum or mean across time of encoder output
            max_length = max(query_lengths)
            comparator = query_lengths[:, None].to(query_enc.device)
            query_mask = torch.arange(max_length)[None, :].to(query_enc.device) < comparator

            while query_mask.ndim < query_enc.ndim:
                # Used in the case of "exhaustive" training
                query_mask = query_mask.unsqueeze(-1)
            query_enc = query_enc * query_mask
            if self.summarizer == 'mean':
                query_enc = query_enc.sum(dim=1) / query_mask.sum(dim=1)
            else:
                query_enc = query_enc.sum(dim=1)
        return {'features': query_enc, 'lengths': query_lengths}

    def encode_document(self, document, **kwargs):
        return self.document_encoder(document, **kwargs)

    def encode(self, query, document, **kwargs):
        query_enc = self.encode_query(query)
        document_enc = self.encode_document(document, **kwargs)
        return query_enc, document_enc

    def search(self, query_enc, document_enc, exhaustive=False, **kwargs):
        if exhaustive:
            search_result = torch.matmul(query_enc['features'],
                                         document_enc['features'].transpose(-1, -2)).squeeze()
        else:
            search_result = torch.matmul(document_enc['features'],
                                         query_enc['features'].unsqueeze(-1)).squeeze()
        search_result = search_result.sigmoid()
        return {'features': search_result, 'lengths': document_enc['lengths']}

    def save(self, outdir):
        params_dict = {'query_sum': self.query_sum,
                       'summarizer': self.summarizer,}
        with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
            _kind.write(self.model_name)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            json.dump(params_dict, _json)
        for attr in self.things_to_save:
            getattr(self, attr).save(os.path.join(outdir, attr))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None):
        params_dict = {'query_sum': False,
                       'summarizer': 'sum'}
        try:
            with open(os.path.join(nnetdir, 'nnet.json')) as _json:
                params_dict_from_file = json.load(_json)
            params_dict.update(params_dict_from_file)
        except FileNotFoundError:
            pass

        # Should be either 'CompositeModel' or 'MultitaskCompositeModel'
        encoder_model_name = open(os.path.join(nnetdir,
                                               'query_encoder',
                                               'nnet_kind.txt')).read().strip()
        encoder_kind = _model_names[encoder_model_name]

        query_encoder = CompositeModel.load_from_dir(os.path.join(nnetdir, 'query_encoder'),
                                                     map_location=map_location)
        encoder_model_name = open(os.path.join(nnetdir,
                                               'document_encoder',
                                               'nnet_kind.txt')).read().strip()
        encoder_kind = _model_names[encoder_model_name]
        document_encoder = encoder_kind.load_from_dir(os.path.join(nnetdir, 'document_encoder'),
                                                      map_location=map_location)
        net = cls(query_encoder, document_encoder, query_sum=params_dict['query_sum'])
        net.to(map_location)
        return net



_model_names = {'CompositeModel': CompositeModel,
                'MultitaskCompositeModel': MultitaskCompositeModel,
                }

_search_model_names = {'SimpleSearchNet': SimpleSearchNet,
                       'SSN': SimpleSearchNet,
                       }


def get_model(model_name, searcher=False):
    if searcher:
        return _search_model_names[model_name]
    else:
        return _model_names[model_name]
