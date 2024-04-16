import os
import json
import time
import torch
from torch import optim
import models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import reduce_lr, stop_early, since


_optimizer_kinds = {'Adam': optim.Adam,
                    'AdamW': optim.AdamW,
                    'SGD': optim.SGD,
                    }


class SimpleLoader:
    def initialize_args(self, **kwargs):
        for key, val in kwargs.items():
            self.params_dict[key] = val
        for key, val in self.params_dict.items():
            setattr(self, key, val)


class SimpleTrainingLoop(SimpleLoader):
    def __init__(self, model, device,
                 pretrained_query_encoder=None,
                 pretrained_document_encoder=None,
                 **kwargs):
        self.params_dict = {
            'optimizer': 'Adam',
            'learning_rate': 2e-4,
            'num_epochs': 200,
            'verbose': True,
            'use_early_stopping': True,
            'early_stopping_loss': 0,
            'cooldown': 0,
            'num_epochs_early_stopping': 10,
            'delta_early_stopping': 1e-4,
            'learning_rate_lower_bound': 1e-6,
            'learning_rate_scale': 0.5,
            'num_epochs_reduce_lr': 4,
            'num_epochs_cooldown': 8,
            'use_model_checkpoint': True,
            'model_checkpoint_period': 1,
            'start_epoch': 0,
            'gradient_clip': None,
            'steps_per_update': 1,
            'train_document_encoder': -1,
            'train_query_encoder': -1,
            'keep_pretrained_embeddings': False,
            'momentum': 0.99,  # For SGD
            'weight_decay': 0.0,  # For AdamW
        }
        super(SimpleTrainingLoop, self).initialize_args(**kwargs)
        self.device = device
        if isinstance(model, str):
            # In this case it's just a path to a previously stored model
            modelsubdir = 'last'
            modeldir = os.path.join(model, modelsubdir)
            self.load_from_dir(modeldir)
            if os.path.isfile(os.path.join(modeldir, 'loss_history.json')):
                print(f"Resuming from epoch {self.elapsed_epochs()}")
            else:
                raise FileNotFoundError(os.path.join(model, '{{nnetdir}}', 'loss_history.json'))
        else:
            self.model = model

            # Load pretrained weights if specified
            if pretrained_query_encoder is not None:
                model_kind_file = os.path.join(pretrained_query_encoder, 'nnet_kind.txt')
                if os.path.isfile(model_kind_file):
                    with open(model_kind_file) as _kind:
                        model_kind = models.get_model(_kind.read())
                else:
                    model_kind = type(self.model.query_encoder)
                pretrained_query_encoder = model_kind.load_from_dir(pretrained_query_encoder,
                                                                    map_location=self.device)
                num_layers = len(pretrained_query_encoder.layers)
                if self.params_dict.get('keep_pretrained_embeddings'):
                    layers_to_load = range(num_layers)
                else:
                    layers_to_load = range(1, num_layers)
                self.model.query_encoder.load_from_other_composite_model(pretrained_query_encoder,
                                                                         layers_to_load,
                                                                         layers_to_load)

            if pretrained_document_encoder is not None:
                model_kind_file = os.path.join(pretrained_document_encoder, 'nnet_kind.txt')
                if os.path.isfile(model_kind_file):
                    with open(model_kind_file) as _kind:
                        model_kind = models.get_model(_kind.read())
                else:
                    model_kind = type(self.model.document_encoder)
                self.model.document_encoder = model_kind.load_from_dir(pretrained_document_encoder,
                                                                       map_location=self.device)

            self.optimizer = self.initialize_optimizer()
            self.loss_history = {}

    def initialize_optimizer(self, load_from=None):
        _opt = _optimizer_kinds[self.params_dict['optimizer']]
        if 'SGD' in self.params_dict['optimizer']:
            optimizer = _opt(self.model.parameters(),
                             lr=self.params_dict['learning_rate'],
                             momentum=self.params_dict['momentum'])
        elif self.params_dict['optimizer'] == "AdamW":
            optimizer = _opt(self.model.parameters(),
                             lr=self.params_dict['learning_rate'],
                             weight_decay=self.params_dict['weight_decay'])
        else:
            optimizer = _opt(self.model.parameters(),
                             lr=self.params_dict['learning_rate'])
        if load_from is not None:
            optimizer.load_state_dict(load_from.state_dict())
        return optimizer

    def unpack_batch(self, batch_data):
        for outer_key, outer_value in batch_data.items():
            if 'dataset' in outer_key:
                continue
            for inner_key, inner_value in outer_value.items():
                batch_data[outer_key][inner_key] = inner_value.to(self.device)

        query = batch_data['query']
        document = batch_data['document']
        label = batch_data['label']

        query['features'] = query['features'][:, :query['lengths'].max()]

        document['features'] = document['features'][:, :document['lengths'].max()]

        if 'query_dataset' in batch_data:
            query['dataset'] = batch_data['query_dataset']
        if 'dataset' in batch_data:
            document['dataset'] = batch_data['dataset']

        label['features'] = label['features'][:, :label['lengths'].max()]
        if 'weights' in label:
            label['weights'] = label['weights'][:, :label['lengths'].max()]
        return query, document, label

    def batch_to_metrics(self, batch_data, criterion, pad_value):
        self.optimizer.zero_grad()
        query, document, label = self.unpack_batch(batch_data)

        search_result = self.model(query, document)

        subs_factor = self._model.document_encoder.get_subsampling_factor()
        label['features'] = label['features'][:, ::subs_factor]
        label['lengths'] = search_result['lengths']
        if 'weights' in label:
            label['weights'] = label['weights'][:, ::subs_factor]

        loss = criterion(search_result['features'], label['features'], sample_weights=label['weights'])

        # Not part of the graph, just metrics to be monitored
        label_batch = label['features']
        detached_search_result = search_result['features'].detach()
        mask = (label['features'] != pad_value).float()

        pred_class = torch.sign(torch.clamp(detached_search_result - 0.5, 0))
        correctness = (pred_class == label_batch).float() * mask
        label_ones = (label_batch == 1).float() * mask  # * sample_weights_batch
        label_zeros = (label_batch == 0).float() * mask  # * sample_weights_batch
        numel = torch.sum(mask).item()
        running_loss = loss.item()  # * numel  # / doc_batch.size(0)  # * (x.size(1) * x.size(2))
        running_class0_sum = (label_zeros * detached_search_result).sum().item()
        running_class1_sum = (label_ones * detached_search_result).sum().item()

        running_class0_acc = (label_zeros * correctness).sum().item()
        running_class1_acc = (label_ones * correctness).sum().item()

        running_class0_count = label_zeros.sum().item()
        running_class1_count = label_ones.sum().item()

        running_count = numel

        metrics = (running_loss,
                   running_class0_sum,
                   running_class1_sum,
                   running_class0_acc,
                   running_class1_acc,
                   running_class0_count,
                   running_class1_count,
                   running_count,)
        return loss, metrics

    @property
    def _model(self):
        return self.model

    @property
    def is_main_process(self):
        return True

    def train_one_epoch(self, criterion, data_loaders, phases=('train', 'test'),
                        **kwargs):
        epoch_beg = time.time()
        self.model.to(self.device)
        dataset_sizes = {x: len(data_loaders[x])
                         for x in phases}

        if self.elapsed_epochs() < self.params_dict['train_query_encoder']:
            self._model.query_encoder.requires_grad_(False)
        if self.elapsed_epochs() < self.params_dict['train_document_encoder']:
            self._model.document_encoder.requires_grad_(False)
        self.loss_history[str(self.elapsed_epochs())] = {}
        if self.is_main_process:
            print('Epoch {}/{} - lr={}'.format(self.elapsed_epochs(), self.params_dict['num_epochs'],
                                               self.optimizer.param_groups[0]['lr'])
                  )
        for phase in phases:
            phase_beg = time.time()
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()
            running_loss = 0.
            running_class0_sum = 0.
            running_class1_sum = 0.
            running_class0_acc = 0.
            running_class1_acc = 0.
            running_class0_count = 0.
            running_class1_count = 0.
            running_count = 0.
            for batch_no, batch_data in enumerate(data_loaders[phase]):
                if (batch_no - 1) % self.params_dict['steps_per_update'] == 0 or batch_no == 0:
                    self.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    total_loss, metrics = self.batch_to_metrics(batch_data, criterion,
                                                                data_loaders[phase].dataset.pad_value)
                    metrics = self.synchronize(metrics)
                running_metrics = (running_loss,
                                   running_class0_sum,
                                   running_class1_sum,
                                   running_class0_acc,
                                   running_class1_acc,
                                   running_class0_count,
                                   running_class1_count,
                                   running_count)
                running_metrics = [a + b for a, b in zip(running_metrics, metrics)]
                (running_loss,
                 running_class0_sum,
                 running_class1_sum,
                 running_class0_acc,
                 running_class1_acc,
                 running_class0_count,
                 running_class1_count,
                 running_count) = running_metrics
                class0_acc = running_class0_acc / running_class0_count
                class1_acc = running_class1_acc / running_class1_count
                class0_sum = running_class0_sum / running_class0_count
                class1_sum = running_class1_sum / running_class1_count
                if phase == 'train':
                    total_loss.backward()
                    if self.params_dict['gradient_clip']:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                       self.params_dict['gradient_clip'])
                    if batch_no % self.params_dict['steps_per_update'] == 0 or batch_no >= len(
                            data_loaders[phase].dataset) - 1:
                        self.optimizer.step()
                phase_elapse = since(phase_beg)
                eta = int(phase_elapse
                          * (dataset_sizes[phase]
                             - batch_no - 1)
                          / (batch_no + 1))
                if self.params_dict['verbose'] and self.is_main_process:
                    print('\r\t{} batch: {}/{} batches - ETA: {}s - loss: {:.4f} - acc: {:.4f} - sep: {:.4f}'.format(
                        phase.title(),
                        batch_no + 1,
                        dataset_sizes[phase],
                        eta, running_loss / running_count,
                        (class0_acc + class1_acc) / 2,
                        class1_sum - class0_sum
                    ), end='')
            epoch_loss = running_loss / running_count
            if self.is_main_process:
                print(" - loss: {:.4f} - acc: {:.4f} - sep: {:.4f}".format(epoch_loss, (class0_acc + class1_acc) / 2,
                                                                           class1_sum - class0_sum))
            self.loss_history[str(self.elapsed_epochs() - 1)][phase] = (epoch_loss,
                                                                        class0_acc, class1_acc,
                                                                        class0_sum, class1_sum,
                                                                        running_class0_count, running_class1_count)
            if hasattr(data_loaders[phase], 'current_epoch'):
                data_loaders[phase].current_epoch += 1
        if self.is_main_process:
            print('\tTime: {}s'.format(int(since(epoch_beg))))
        self._model.query_encoder.requires_grad_(True)
        self._model.document_encoder.requires_grad_(True)

    def train(self, outdir, criterion, data_loaders,
              phases=('train', 'test'),
              job_num_epochs=None,
              **kwargs):
        if not job_num_epochs:
            job_num_epochs = self.params_dict['num_epochs']

        for i in range(self.params_dict['num_epochs']):
            self.train_one_epoch(criterion, data_loaders, phases, **kwargs)
            self.save_to_dir(os.path.join(outdir, 'last'))

            if self.params_dict['use_model_checkpoint'] \
                    and (self.elapsed_epochs() % self.params_dict['model_checkpoint_period'] == 0):
                self.save_to_dir(os.path.join(outdir,
                                              str(self.elapsed_epochs())))
            history_sum = [self.loss_history[str(_)][phases[-1]][self.params_dict['early_stopping_loss']]
                           for _ in range(self.elapsed_epochs())]
            if history_sum[-1] == min(history_sum):
                self.save_to_dir(os.path.join(outdir, 'best'))
            rl = reduce_lr(history=history_sum,
                           lr=self.optimizer.param_groups[0]['lr'],
                           cooldown=self.params_dict['cooldown'],
                           patience=self.params_dict['num_epochs_reduce_lr'],
                           mode='min',
                           difference=self.params_dict['delta_early_stopping'],
                           lr_scale=self.params_dict['learning_rate_scale'],
                           lr_min=self.params_dict['learning_rate_lower_bound'],
                           cool_down_patience=self.params_dict['num_epochs_cooldown'])
            self.optimizer.param_groups[0]['lr'], self.params_dict['cooldown'] = rl
            if self.params_dict['use_early_stopping']:
                if stop_early(history_sum,
                              patience=self.params_dict['num_epochs_early_stopping'],
                              mode='min',
                              difference=self.params_dict['delta_early_stopping']):
                    print('Stopping Early.')
                    break
            if self.elapsed_epochs() >= self.params_dict['num_epochs']:
                break
            if i >= job_num_epochs:
                return
        with open(os.path.join(outdir, '.done.train'), 'w') as _w:
            pass

    def elapsed_epochs(self):
        return len(self.loss_history)

    def load_from_dir(self, trainer_dir, model_kind=models.CompositeModel):
        if os.path.isfile(os.path.join(trainer_dir, 'nnet_kind.txt')):
            model_classname = open(os.path.join(trainer_dir, 'nnet_kind.txt')).read().strip()
            model_kind = models.get_model(model_classname, searcher=True)
        self.model = model_kind.load_from_dir(trainer_dir)
        self.model.to(self.device)
        with open(os.path.join(trainer_dir, 'optimizer.txt')) as _opt:
            opt_name = _opt.read()
        _opt = _optimizer_kinds[opt_name]
        self.optimizer = _opt(self.model.parameters(), lr=1e-3)
        self.optimizer.load_state_dict(torch.load(
            os.path.join(trainer_dir, 'optimizer.state'))
        )
        jsonfile = os.path.join(trainer_dir, 'trainer.json')
        with open(jsonfile) as _json:
            self.params_dict = json.load(_json)
        jsonfile = os.path.join(trainer_dir, 'loss_history.json')
        with open(jsonfile) as _json:
            self.loss_history = json.load(_json)

    def save_to_dir(self, trainer_dir):
        if self.is_main_process:
            if not os.path.isdir(trainer_dir):
                os.makedirs(trainer_dir)
            self._model.save(trainer_dir)
            opt_name = str(self.optimizer).split()[0]
            with open(os.path.join(trainer_dir, 'optimizer.txt'), 'w') as _opt:
                _opt.write(opt_name)
            torch.save(self.optimizer.state_dict(),
                    os.path.join(trainer_dir, 'optimizer.state')
                    )
            with open(os.path.join(trainer_dir, 'trainer.json'), 'w') as _json:
                json.dump(self.params_dict, _json)
            with open(os.path.join(trainer_dir, 'loss_history.json'), 'w') as _json:
                json.dump(self.loss_history, _json)

    def synchronize(self, metrics):
        return metrics


class DistributedDataParallelTrainingLoop(SimpleTrainingLoop):
    @property
    def _model(self):
        return self.model.module

    @property
    def is_main_process(self):
        return int(os.environ["RANK"]) == 0

    def train(self, outdir, criterion, data_loaders,
              phases=('train', 'test'),
              job_num_epochs=None,
              **kwargs):
        self.model.to(self.device)
        self.model = DDP(self.model, find_unused_parameters=True)
        self.optimizer = self.initialize_optimizer(load_from=self.optimizer)
        super().train(
            outdir=outdir,
            criterion=criterion,
            data_loaders=data_loaders,
            phases=phases,
            job_num_epochs=job_num_epochs,
            **kwargs,
            )

    def synchronize(self, metrics):
        sync_metrics = torch.tensor(metrics).cuda()
        dist.all_reduce(sync_metrics)
        return sync_metrics.tolist()


_trainers = {'Simple': SimpleTrainingLoop,
             'Elastic': DistributedDataParallelTrainingLoop,
             }


def get_trainer(trainer_name):
    return _trainers[trainer_name]
