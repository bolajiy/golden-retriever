import math
import os
import abc
import json
import torch
from torch import nn
from torchaudio.models import conformer
import utils
from typing import Any, Dict, Iterable, List, Union


class GenericLayer(abc.ABC, nn.Module):
    '''
    Neural network layers which take with BxNxD input and BxN'xD output.
    '''
    def __init__(self, previous_layer=None, input_dim=None, output_dim=None, samedim: bool = False,
                 **kwargs):
        super().__init__()
        self.model_name = 'GenericLayer'
        if input_dim is None:
            input_dim = previous_layer.get_output_dim()
        self.input_dim = input_dim
        if not isinstance(self.input_dim, int):
            raise ValueError(f'input_dim should be int got {type(self.input_dim)} instead')
        self.output_dim = output_dim
        if samedim:  # e.g for dropout and batchnorm
            self.output_dim = self.input_dim
        if not isinstance(self.output_dim, int):
            raise ValueError(f'output_dim should be int got {type(self.output_dim)} instead')
        self.loss_cache = {}

    def get_output_dim(self):
        return self.output_dim

    def get_subsampling_factor(self):
        return 1

    def get_output_lengths(self, input_lengths):
        return [int((ln+self.get_subsampling_factor() - 1) / self.get_subsampling_factor()) for ln in input_lengths]


class GRULayer(GenericLayer):
    '''Stack of GRU layers'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None,
                 bidirectional: bool = False,
                 num_layers: int = 1,
                 dropout: float = 0):
        super().__init__(previous_layer, input_dim, output_dim)
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(self.input_dim, self.output_dim, num_layers=num_layers, bidirectional=self.bidirectional,
                              batch_first=True, dropout=dropout)

    def forward(self, x: Dict[str, Any]):
        features, _ = self.encoder(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y

    def get_output_dim(self):
        ndir = 2 if self.bidirectional else 1
        return ndir * self.output_dim

    def flatten(self):
        self.encoder.flatten_parameters()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.flatten()


class LSTMLayer(GRULayer):
    '''Stack of LSTM layers'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None,
                 bidirectional: bool = False,
                 num_layers: int = 1,
                 dropout: float = 0):
        super(GRULayer, self).__init__(previous_layer, input_dim, output_dim)
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(self.input_dim, self.output_dim, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True, dropout=dropout)


class FeedForwardLayer(GenericLayer):
    '''Single feedforward layer'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None):
        super().__init__(previous_layer, input_dim, output_dim)
        self.output_dim = output_dim
        self.encoder = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x: Dict[str, Any]):
        features: torch.Tensor = self.encoder(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y


class CNNLayer(GenericLayer):
    '''Single 1D CNN layer'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None, kernel_size: int = 5, dilation: int = 2,
                 stride: int = 1):
        super().__init__(previous_layer, input_dim, output_dim)
        if not kernel_size % 2:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = int(dilation * (kernel_size - 1) / 2)
        self.stride = stride
        self.encoder = nn.Conv1d(self.input_dim, self.output_dim, kernel_size=self.kernel_size, dilation=self.dilation,
                                 stride=self.stride, padding=self.padding)

    def forward(self, x: Dict[str, Any]):
        features = x['features'].permute(0, 2, 1)
        features = self.encoder(features)
        features = features.permute(0, 2, 1)
        y = {k: v for k, v in x.items()}
        y['features'] = features
        # if x.get('lengths'):
        y['lengths'] = self.get_output_lengths(x['lengths'])
        return y

    def get_subsampling_factor(self):
        return self.stride


class CNN2DLayer(GenericLayer):
    '''Single 2D CNN layer'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None, kernel_size: int = 3, dilation: int = 1,
                 stride: int = 1):
        if input_dim is None:
            input_dim = previous_layer.get_output_dim()
        self.num_filters = int(output_dim * stride / input_dim)
        output_dim = int(self.num_filters * input_dim / stride)
        super().__init__(previous_layer, input_dim, output_dim)
        if not kernel_size % 2:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = int(dilation * (kernel_size - 1) / 2)
        self.encoder = nn.Conv2d(1, self.num_filters, kernel_size=self.kernel_size,
                                 dilation=self.dilation, padding=self.padding, stride=self.stride)

    def forward(self, x: Dict[str, Any]):
        features = x['features'].unsqueeze(1)
        features = self.encoder(features)
        b, c, n, w = features.size()
        features = features.transpose(1, 2)
        features = features.contiguous().view(b, n, w * c)
        y = {k: v for k, v in x.items()}
        y['features'] = features
        # if x.get('lengths'):
        y['lengths'] = self.get_output_lengths(x['lengths'])
        return y

    def get_subsampling_factor(self):
        return self.stride


class CNN2DMultiLayer(GenericLayer):
    '''Stack of 2D CNN layers'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 output_dim: int = None, kernel_size: int = 3, dilation: int = 1,
                 stride: int = 1, num_layers=1):
        total_stride = 1
        for _ in range(num_layers):
            total_stride *= stride
        if input_dim is None:
            input_dim = previous_layer.get_output_dim()
        num_filters = int(output_dim * total_stride / input_dim)
        output_dim = int(num_filters * input_dim / total_stride)
        super().__init__(previous_layer, input_dim, output_dim)
        if not kernel_size % 2:
            kernel_size += 1
        self.num_layers = num_layers
        self.total_stride = total_stride
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = int(dilation * (kernel_size - 1) / 2)
        input_filters = [self.num_filters for _ in range(num_layers)]
        input_filters[0] = 1
        self.encoder = nn.ModuleList([nn.Conv2d(inf, self.num_filters, kernel_size=self.kernel_size,
                                                dilation=self.dilation, padding=self.padding, stride=self.stride)
                                      for inf in input_filters])

    def forward(self, x: Dict[str, Any]):
        features = x['features'].unsqueeze(1)
        for layer in self.encoder:
            features = layer(features)
            features = nn.ReLU()(features)
        features = features.transpose(1, 2)
        b, n, c, w = features.size()
        features = features.contiguous().view(b, n, w * c)
        y = {k: v for k, v in x.items()}
        y['features'] = features
        # if x.get('lengths'):
        y['lengths'] = self.get_output_lengths(x['lengths'])
        return y

    def get_subsampling_factor(self):
        return self.total_stride


class PositionalEncoding(nn.Module):
    '''Vanilla transformer positional encoding'''
    # Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def lengths_to_mask(lengths: torch.Tensor):
    if lengths is None:
        return None
    max_length = lengths.max()
    mask = torch.arange(max_length)[None, :].to(lengths.device) >= lengths[:, None]
    return mask


class TransformerLayer(GenericLayer):
    '''Stack of transformer encoder layers'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 num_layers: int = 6, nhead: int = 8, max_len: int = 1000,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 norm_first=False, use_pe: bool = True):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.nhead = nhead
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.get_output_dim(), nhead=self.nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   norm_first=norm_first,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        if use_pe:
            self.positional_encoding = PositionalEncoding(self.get_output_dim(), max_len=max_len)
        else:
            self.positional_encoding = None

    def forward(self, x: Dict[str, Any]):
        if self.positional_encoding is None:
            features = x['features']
        else:
            features = self.positional_encoding(x['features'])
        mask = lengths_to_mask(x.get('lengths'))
        features = self.encoder(features, src_key_padding_mask=mask)
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y



class ConformerLayer(GenericLayer):
    '''Stack of conformer layers'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 num_layers: int = 12, nhead: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 depthwise_conv_kernel_size=13, use_group_norm=True, convolution_first=False,):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder = conformer.Conformer(
            input_dim=self.get_output_dim(),
            num_heads=nhead,
            ffn_dim=dim_feedforward,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )

    def forward(self, x: Dict[str, Any]):
        features = x['features']
        lengths = x.get('lengths')
        if lengths is None:
            b, l, _ = features.shape
            lengths = torch.zeros(b, dtype=torch.int, device=x.device) + l
        y, lengths = self.encoder(x, lengths)
        y = {k: v for k, v in x.items()}
        y['features'] = features


class EmbeddingLayer(GenericLayer):
    '''Embedding layer for discrete inputs'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, output_dim: int = None,
                 pad=False, padding_idx=None):
        super().__init__(None, input_dim, output_dim)
        if pad and padding_idx is None:
            # In this case assume the last embedding is used for padding
            padding_idx = input_dim
        # In theory the line below should be within an `if pad:` block but is left outside for convenience so that
        # if `padding_idx` is set, it will behave as if `pad` is True regardless of the actual value of `pad`
        self.encoder = nn.Embedding(self.input_dim, self.output_dim, padding_idx=padding_idx)

    def forward(self, x: Dict[str, Any]):
        features = self.encoder(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y

    def override(self, source_embeddings, source_dims=None, target_dims=None):
        if source_dims is None:
            source_dims = list(range(self.input_dim))
        if target_dims is None:
            target_dims = list(range(self.input_dim))
        if not isinstance(source_embeddings, nn.Embedding):
            # In case it's another EmbeddingLayer object
            source_embeddings = source_embeddings.encoder
        self.encoder.weight.data[target_dims] = source_embeddings.weight.data[source_dims].data.clone()


class DropoutLayer(GenericLayer):
    '''Dropout layer'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, p=0.):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.encoder = nn.Dropout(p)

    def forward(self, x: Dict[str, Any]):
        features = self.encoder(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y


class ReLULayer(GenericLayer):
    '''Relu layer, which also doubles as generic activation layer'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, override_to_activation: str = None,
                 **kwargs):
        super().__init__(previous_layer, input_dim, samedim=True)
        if override_to_activation is None:
            self.encoder = nn.ReLU()
        else:
            # A very ugly way of supporting other activation functions without having to write them manually
            # e.g. by setting override_to_activation='Tanh'
            # Use at your own peril
            self.encoder = getattr(nn, override_to_activation)(**kwargs)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        return y


class SubsamplingLayer(GenericLayer):
    '''Downsampling layer. Supports frame skipping, concatenation or summation'''
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None, factor=1, concat=False, use_sum=False):
        super().__init__(previous_layer, input_dim, samedim=True)
        self.subsampling_factor = factor
        self.concat = concat
        self.use_sum = use_sum
        assert not (concat and use_sum), "only one of concat and use_sum can be set to True"
        if self.concat:
            self.output_dim = self.subsampling_factor * self.output_dim
            self.encoder = lambda x: [x[:, i::self.subsampling_factor] for i in range(self.subsampling_factor)]
        elif self.use_sum:
            self.encoder = lambda x: [x[:, i::self.subsampling_factor] for i in range(self.subsampling_factor)]
        else:
            self.encoder = lambda x: x[:, ::self.subsampling_factor]

    def forward(self, x: torch.Tensor):
        def _forward(inner_x):
            if self.concat:
                inner_x = self.encoder(inner_x)
                lens = [_.size(1) for _ in inner_x]
                maxlen = max(lens)
                inner_x = [torch.cat((arr, torch.zeros(inner_x[0].size(0), maxlen - lv, inner_x[0].size(-1), device=arr.device)), dim=1)
                           for lv, arr in zip(lens, inner_x)]
                inner_x = torch.cat(inner_x, dim=-1)
            elif self.use_sum:
                inner_x = self.encoder(inner_x)
                lens = [_.size(1) for _ in inner_x]
                maxlen = max(lens)
                inner_x = [torch.cat((arr, torch.zeros(inner_x[0].size(0), maxlen - lv, inner_x[0].size(-1), device=arr.device)), dim=1)
                           for lv, arr in zip(lens, inner_x)]
                return sum(inner_x)
            else:
                inner_x = self.encoder(inner_x)
            return inner_x
    
        features = _forward(x['features'])
        y = {k: v for k, v in x.items()}
        y['features'] = features
        # if x.get('lengths'):
        y['lengths'] = self.get_output_lengths(x['lengths'])
        return y

    def get_subsampling_factor(self):
        return self.subsampling_factor


class ResidualLayer(GenericLayer):
    def __init__(self, previous_layer: nn.Module = None, input_dim: int = None,
                 meta_layer_name: str = None, input_reshape=False, output_reshape=False, **kwargs):
        meta_layer_dict = {'layer_name': meta_layer_name}
        meta_layer = make_layer(meta_layer_dict, previous_layer=previous_layer, input_dim=input_dim, **kwargs)
        input_dim = meta_layer.input_dim
        output_dim = meta_layer.get_output_dim()
        if output_reshape:
            output_reshape_layer = nn.Linear(output_dim, input_dim)
        else:
            output_reshape_layer = None
        if input_reshape:
            input_reshape_layer = nn.Linear(input_dim, output_dim)
        else:
            input_reshape_layer = None
        if not (input_reshape or output_reshape):
            assert input_dim == output_dim, (f"input and output dimensions should be equal, got {input_dim}"
                                             f" and {output_dim} instead, use one of the reshape flags")
        assert not (input_reshape and output_reshape), "only one of input_reshape and output_reshape should be set"
        super().__init__(previous_layer, input_dim, output_dim=output_dim, samedim=(not input_reshape))
        self.meta_layer = meta_layer
        self.input_reshape = input_reshape
        self.output_reshape = output_reshape
        self.input_reshape_layer = input_reshape_layer
        self.output_reshape_layer = output_reshape_layer
        self.subsampling_factor = meta_layer.get_subsampling_factor()
        assert self.subsampling_factor == 1

    def forward(self, x: torch.Tensor):
        features = self.meta_layer(x['features'])
        if self.output_reshape:
            features = self.output_reshape_layer(features)
        if self.input_reshape:
            x['features'] = self.input_reshape_layer('features')
        
        y = {k: v for k, v in x.items()}
        y['features'] = features + x['features']
        return x + y


class CompositeModel(nn.Module):
    '''Stack of GenericLayers. Saves and loads model configuration in jsons
    '''
    def __init__(self, layers_dict, ordering=None, input_dim=None, output_dim=None):
        super().__init__()
        self.model_name = 'CompositeModel'
        if isinstance(layers_dict, str):
            with open(layers_dict) as _json:
                layers_dict = json.load(_json)
            ordering = layers_dict['ordering']
            layers_dict = layers_dict['layers']
        self.layers_dict = layers_dict
        if ordering is None:
            ordering = sorted([int(x) for x in layers_dict.keys()])
        self.ordering = [str(x) for x in ordering]
        if input_dim is not None:
            self.layers_dict[self.ordering[0]]['input_dim'] = input_dim
        if output_dim is not None:
            self.layers_dict[self.ordering[-1]]['output_dim'] = output_dim
        layers = [make_layer(self.layers_dict[self.ordering[0]])]
        self.input_dim = layers[0].input_dim
        for key in self.ordering[1:]:
            layer = make_layer(self.layers_dict[key], previous_layer=layers[-1])
            layers.append(layer)
        self.output_dim = layers[-1].get_output_dim()
        self.layers = nn.ModuleList(layers)
        print(self.layers)
        print(f'total_params: {sum([p.numel() for p in self.parameters()])}')

    def summarize(self, input_sequence, mask=None, dim=1, **kwargs):
        x = self(input_sequence, **kwargs)
        if mask is not None:
            if mask.ndim < x.ndim:
                mask = mask.unsqueeze(-1).expand(*x.shape)
            x = x * mask
        return x.sum(dim=dim)

    def forward(self, x: Dict[str, Any], layers_to_use=None):
        if layers_to_use is None:
            for layer in self.layers:
                x = layer(x)
        else:
            for layer_id in layers_to_use:
                layer = self.layers[layer_id]
                x = layer(x)
        return x

    def get_subsampling_factor(self):
        s = 1
        for layer in self.layers:
            s *= layer.get_subsampling_factor()
        return s

    def save(self, outdir):
        utils.chk_mkdir(outdir)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _json:
            savedict = {'layers': self.layers_dict,
                        'ordering': self.ordering}
            json.dump(savedict, _json)
        if hasattr(self, 'model_name'):
            with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
                _kind.write(self.model_name)
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))

    def load_from_other_composite_model(self, other_composite_model: nn.Module,
                                        source_layer_ids: list = None,
                                        target_layer_ids: list = None,
                                        clone=False):
        if source_layer_ids is None:
            source_layer_ids = list(range(len(self.layers)))
        if target_layer_ids is None:
            target_layer_ids = source_layer_ids
        assert len(target_layer_ids) == len(source_layer_ids)
        source_layers = [layer for layer in other_composite_model.layers]
        for s, t in zip(source_layer_ids, target_layer_ids):
            if clone:
                self.layers[t] = source_layers[s]
            else:
                self.layers[t].load_state_dict(source_layers[s].state_dict())
            print(f'loaded layer {self.layers[t]}')
        self.layers = nn.ModuleList([layer for layer in self.layers])

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None,
                      random_init_model=None,
                      source_layer_ids=None,
                      target_layer_ids=None, ):
        net = cls(os.path.join(nnetdir, 'nnet.json'))
        state_dict = torch.load(os.path.join(nnetdir, 'nnet.mdl'),
                                map_location=map_location)
        net.to(map_location)
        net.load_state_dict(state_dict)
        if random_init_model is not None:
            net.load_from_other_composite_model(random_init_model,
                                                source_layer_ids,
                                                target_layer_ids,
                                                )
        return net


class MultitaskCompositeModel(nn.Module):
    '''Set of CompositeModels for input from various modalities (e.g. text and speech), where each input defines its own pathway '''
    def __init__(self, model_dict: Union[str, dict], input_dims=None):
        super().__init__()
        if input_dims is None:
            input_dims = {}
        if isinstance(model_dict, str):
            with open(model_dict) as _inp:
                model_dict: dict = json.load(_inp)
        self.model_name = 'MultitaskCompositeModel'
        self.model_dict = model_dict

        self.tasks: List[str] = self.model_dict['tasks']
        self.default_task: str = self.model_dict['default_task']
        self.models_per_task: Dict[str, List[int]] = self.model_dict['models_per_task']
        sub_models = model_dict['sub_models']

        input_dims = [input_dims.get(i) for i in range(len(sub_models))]
        sub_models: List[CompositeModel] = [CompositeModel(mdl['layers'], ordering=mdl['ordering'], input_dim=dim)
                                            for dim, mdl in zip(input_dims, sub_models)]
        self.sub_models = nn.ModuleList(sub_models)

        assert self.default_task in self.tasks
        for task in self.tasks:
            assert task in self.models_per_task.keys(), f"module for {task} not found"

        self.pretrained_models_definitions: Dict[int, dict] = model_dict.get('pretrained_models_definitions')
        if self.pretrained_models_definitions is not None:
            self.pretrained_models_definitions = {int(k): v for k, v in self.pretrained_models_definitions.items()}
            for key in self.pretrained_models_definitions.keys():
                assert key in range(len(self.sub_models))
            self.load_pretrained_models(self.pretrained_models_definitions)

        self.pretrained_mt_model_definition: Dict[int, dict] = model_dict.get('pretrained_mt_model_definition')
        if self.pretrained_mt_model_definition is not None:
            self.pretrained_mt_model_definition = {int(k): v for k, v in self.pretrained_mt_model_definition.items()}
            for key in self.pretrained_mt_model_definition.keys():
                assert key in range(len(self.sub_models))
            self.load_pretrained_multitask_model(self.pretrained_mt_model_definition)
        self.input_dim = self.get_input_dims()

    def load_pretrained_models(self, pretrained_models_definitions):
        for key, model_def in pretrained_models_definitions.items():
            model_dir = model_def['directory']
            pretrained_model = CompositeModel.load_from_dir(model_dir)
            source_layer_ids = model_def.get('source_layer_ids')
            target_layer_ids = model_def.get('target_layer_ids')
            self.sub_models[key].load_from_other_composite_model(pretrained_model,
                                                                 source_layer_ids=source_layer_ids,
                                                                 target_layer_ids=target_layer_ids)

    def load_pretrained_multitask_model(self, pretrained_multitask_model_definition):
        for key, model_def in pretrained_multitask_model_definition.items():
            model_dir = model_def['directory']
            pretrained_model = MultitaskCompositeModel.load_from_dir(model_dir)
            source_layer_ids = model_def.get('source_layer_ids')
            target_layer_ids = model_def.get('target_layer_ids')
            source_sub_model = model_def.get('source_sub_model', key)
            self.sub_models[key].load_from_other_composite_model(pretrained_model.sub_models[source_sub_model],
                                                                 source_layer_ids=source_layer_ids,
                                                                 target_layer_ids=target_layer_ids)

    def get_input_dims(self, task: str = None):
        if task is None:
            task = self.default_task
        model_id = self.models_per_task[task][0]
        first_sub_model = self.sub_models[model_id]
        return first_sub_model.input_dim

    def forward(self, x: Dict[str, Any]):
        task = x.get('dataset')
        # task = x.get('task', task)

        if task is None:
            task = self.default_task
        models_ids = self.models_per_task[task]
        for model_id in models_ids:
            x = self.sub_models[model_id](x)
        return x

    def get_subsampling_factor(self, task: str = None):
        if task is None:
            task = self.default_task
        s = 1
        models_ids = self.models_per_task[task]
        for model_id in models_ids:
            s *= self.sub_models[model_id].get_subsampling_factor()
        return s

    def save(self, outdir, save_submodels=False):
        utils.chk_mkdir(outdir)
        with open(os.path.join(outdir, 'nnet.json'), 'w') as _out:
            json.dump(self.model_dict, _out, indent=4)
        with open(os.path.join(outdir, 'nnet_kind.txt'), 'w') as _kind:
            _kind.write(self.model_name)
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))
        if save_submodels:
            for i, sub_model in enumerate(self.sub_models):
                sub_model.save(os.path.join(outdir, str(i)))

    @classmethod
    def load_from_dir(cls, nnetdir, map_location=None, ):
        model_dict = os.path.join(nnetdir, 'nnet.json')
        net = cls(model_dict)
        net.to(map_location)
        state_dict = torch.load(os.path.join(nnetdir, 'nnet.mdl'),
                                map_location=map_location)
        net.load_state_dict(state_dict)
        return net


_layers = {'GRULayer': GRULayer,
           'GRU': GRULayer,
           'LSTMLayer': LSTMLayer,
           'LSTM': LSTMLayer,
           'FeedForwardLayer': FeedForwardLayer,
           'FF': FeedForwardLayer,
           'CNNLayer': CNNLayer,
           'CNN': CNNLayer,
           'Dropout': DropoutLayer,
           'Drop': DropoutLayer,
           'ReLU': ReLULayer,
           'SubsamplingLayer': SubsamplingLayer,
           'Subs': SubsamplingLayer,
           'EmbeddingLayer': EmbeddingLayer,
           'Embedding': EmbeddingLayer,
           'TransformerLayer': TransformerLayer,
           'Transformer': TransformerLayer,
           'ConformerLayer': ConformerLayer,
           'Conformer': ConformerLayer,
           'CNN2DLayer': CNN2DLayer,
           'CNN2D': CNN2DLayer,
           'CNN2DMultiLayer': CNN2DMultiLayer,
           'CNN2DMulti': CNN2DMultiLayer,
           'ResidualLayer': ResidualLayer,
           'Residual': ResidualLayer,
           }


def make_layer(layer_dict: dict, previous_layer=None, **kwargs) -> GenericLayer:
    layer_dict = layer_dict.copy()
    layer_name = layer_dict.pop('layer_name')
    layer = _layers[layer_name](previous_layer=previous_layer, **layer_dict, **kwargs)
    return layer
