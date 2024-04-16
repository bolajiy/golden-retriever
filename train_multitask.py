import argparse
import os
import utils
import json
import models
import dataloaders
import training
import torch
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
import losses


def main():
    parser = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=parser,
                                     description='RNN based KWS training with paired speech-text and unpaired text')

    parser.add_argument('--no-cuda', action='store_true',
                        help='if specified, turns off gpu usage.')
    parser.add_argument('--distributed-backend', '--backend', default='nccl', choices=['gloo', 'nccl'],
                        help='backend to use for distributed training.')
    parser.add_argument('--resume', action='store_true',
                        help='resume from previous checkpoint')
    parser.add_argument('--num-workers', '--nw', type=int, default=2,
                        help='number of jobs for loading data')

    # Model params
    parser.add_argument('--query-encoder-json', '--qmj', required=True,
                        help='json file defining the query encoder model structure as in layers.py')
    parser.add_argument('--document-encoder-json', '--dmj', required=True,
                        help='json file defining the document encoder model structure as in layers.py')

    parser.add_argument('--pretrained-document-encoder', '--ptd',
                        help='path to a pretrained model which, if specified, will be used to initialize the '
                             'document encoder')
    parser.add_argument('--pretrained-query-encoder', '--ptq',
                        help='path to a pretrained model which, if specified, will be used to initialize the '
                             'query encoder')
    parser.add_argument('--use-query-sum', '--qs', action='store_true',
                        help='use sum of query encoder output across time to represent to the query'
                        'instead of using the output at the last time step')
    parser.add_argument('--query-sum-type', '--qst', default='sum', choices=['sum', 'mean'],
                        help='how to summarize the query')

    # All dataloader params
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--char-map', '--lower',
                        help='colon separated file that contains extra mappings between capital and small letters, '
                             'e.g. "İ": "i", "I": "ı" in Turkish')

    # Paired data daloader params
    parser.add_argument('-m', '--min_ngram', default=1, type=int,
                        help='minimum ngram order for a term.')
    parser.add_argument('-n', '--max_ngram', default=3, type=int,
                        help='maximum ngram order for a term.')
    parser.add_argument('--min-kw-length', '--mkl', type=int, default=3,
                        help='minimum number of letters in a training word, '
                             'words with fewer letters are discarded')
    parser.add_argument('--max-kw-length', '--nkl', type=int, default=1000,
                        help='maximum number of letters in a training word, '
                             'words with more letters are discarded')
    parser.add_argument('--negative-samples-per-doc', '--npd', type=int, default=3,
                        help='number of negative training samples per document')
    parser.add_argument('--validation-split', '--spl', type=float, default=0.1,
                        help='Ratio of data to use of validation')
    parser.add_argument('--utt-max-len', '--uml', type=float, default=1,
                        help='maximum length in seconds of an utterance '
                             'for RNN training')

    # Unpaired text dataloader params
    parser.add_argument('--positive-samples-per-doc-text', '--tppd', type=int, default=1,
                        help='number of positive training samples per text document, '
                        'essentially repeats the same utterance with different masks')
    parser.add_argument('--negative-samples-per-doc-text', '--tnpd', type=int, default=3,
                        help='number of negative training samples per text document')
    parser.add_argument('--text-repetition', '--trep', type=int, default=2,
                        help='number of times to repeat each letter')
    parser.add_argument('--text-use-blank-token', '--tblank', action='store_true',
                        help='use a CTC-style blank token for repetitions instead of repeating each letter', )
    parser.add_argument('--text-use-unk-token', '--tunk', action='store_true',
                        help='use an unk token for unknown characters', )
    parser.add_argument('--mask-probability-text', '--tmask', type=float, default=0.3,
                        help='probability of zeroing out contiguous portions of each training text document')
    parser.add_argument('--text-sampling-weight', '--tsw', type=float, default=1.,
                        help='relative probability of sampling the text task')
    parser.add_argument('--text-loss-weight', '--tlw', type=float, default=1.,
                        help='relative loss of the text task')
    parser.add_argument('--text-negative-samples-weight', '--tnsw', type=float, default=1.,
                        help='relative loss of the negative tokens for the text task',)
    parser.add_argument('--text-max-training-utt-len', '--tuml', type=int, default=500,
                        help='maximum number of tokens in each training utterance', )
    parser.add_argument('--text-max-training-utt-len-word', '--tumlw', type=int,
                        help='maximum number of words in each training utterance', )
    parser.add_argument('--text-force-mask-query', '--tfmq', action='store_true',
                        help='force the query location to be masked', )
    parser.add_argument('--text-force-unmask-query', '--tumq', action='store_true',
                        help='force the query location to be masked', )


    # Loss function params
    parser.add_argument('--zero-weight', '--wz', type=float, default=0.2,
                        help='loss function weight for zero-labeled samples')
    parser.add_argument('--one-weight', '--wo', type=float, default=1.,
                        help='loss function weight for one-labeled samples')
    parser.add_argument('--loss-margin', '--margin', type=float, default=0.7,
                        help='margin to be used in loss computation')

    # Trainer params
    parser.add_argument('--job-num-epochs', '--nj', type=int, default=1,
                        help='number of training epochs for a single job')
    parser.add_argument('--trainer', '--tc', default='SimpleMultitask',
                        help='trainer name as defined in training.py')
    parser.add_argument('--trainer-json', '--tj', '--tconf',
                        help='path to trainer config jsonfile')
    parser.add_argument('--steps-per-epoch', '--spe', type=int, default=10000,
                        help='number of training steps to take for each epoch')

    # Data params
    parser.add_argument('--lexicon',
                        help='optional semi-colon separated mapping from words to phones')
    parser.add_argument('--old-tokenizer',
                        help='path to tokenizer dictionary (or old RttmHolder) pickle '
                             'that will override what is inferred from the rttm file')
    parser.add_argument('--rttm-tolerance', '--tol', type=float, default=0.2)
    parser.add_argument('--frame-length', type=float, default=0.01,
                        help='length of each frame seconds')
    parser.add_argument('--dataloader-kind', '--dkd', default='default', choices=('kaldi', 'mmap', 'wav', 'default'),
                        help='How to load the data directory. '
                             'default loads everything to memory which is ideal for small datasets/dimesions. '
                             'mmap uses a memory map which gets periodically reset to keep memory footprint low '
                             )

    parser.add_argument('--label-type', choices=['rttm', 'ctm'],
                        help='read ctm file instead of rttm')
    
    parser.add_argument('text_file', help='The unpaired training text tsv')
    parser.add_argument('label_file',
                        help='Path to timestamped reference RTTM/CTM file')
    parser.add_argument('datadir', help='The training features')
    parser.add_argument('output_directory', help='Model output directory')
    
    args = parser.parse_args()
    outdir = args.output_directory

    if args.no_cuda:
        dev_name = 'cpu'
    else:
        dev_name = 'cuda:0'

    distributed = torch.distributed.is_torchelastic_launched()
    if distributed:
        torch.distributed.init_process_group(args.distributed_backend)

    device = torch.device(dev_name)

    datadir = args.datadir
    label_file = args.label_file
    
    data = dataloaders.data_utils.UtteranceIterator(datadir, kind=args.dataloader_kind)
    keys = data.utt_dict
    featdim = data[0].shape[-1]

    if not (os.path.isfile(os.path.join(outdir, 'rttm.pkl')) and os.path.isfile(os.path.join(outdir, 'tokenizer.pkl'))):
        run = True
        if distributed and int(os.environ["RANK"]) != 0:
            run = False
        if run:
            print("Creating rttm holder")
            if args.lexicon:
                word2phones = args.lexicon
            else:
                word2phones = None
            if args.old_tokenizer:
                all_phones_override = utils.pkl_load(args.old_tokenizer)
                if not isinstance(all_phones_override, dict):
                    # Assume it's an old RttmHolder
                    all_phones_override = all_phones_override.letter2int
            else:
                all_phones_override = None
            rttm_holder = dataloaders.RttmHolder(label_file, args.min_ngram, args.max_ngram, args.rttm_tolerance,
                                                 args.min_kw_length, args.max_kw_length, allowed_docs=keys.keys(),
                                                 word2phones_file=word2phones, char_map=args.char_map,
                                                 is_ctm=args.label_type == 'ctm',
                                                 all_phones_override=all_phones_override,
                                                 )

            utils.pkl_save(rttm_holder, os.path.join(outdir, 'rttm.pkl'))
            utils.pkl_save(rttm_holder.letter2int, os.path.join(outdir, 'tokenizer.pkl'))

    if distributed:
        # torch.distributed.barrier()
        x = torch.tensor(1.).cuda()
        torch.distributed.all_reduce(x)

    print("Loading rttm holder")
    rttm_holder = utils.pkl_load(os.path.join(outdir, 'rttm.pkl'))
    
    pad_value = -1000

    paired_train_dataset = dataloaders.KwsTrainingDataset(
        data=data,
        keys=keys,
        rttm_holder=rttm_holder,
        negative_samples_per_doc=args.negative_samples_per_doc,
        frame_length=args.frame_length,
        max_length=args.utt_max_len,
        pad_value=pad_value,
        validation_split=args.validation_split,
        validation=False,
    )
    validation_dataset = dataloaders.KwsTrainingDataset(
        data=data,
        keys=keys,
        rttm_holder=rttm_holder,
        negative_samples_per_doc=args.negative_samples_per_doc,
        frame_length=args.frame_length,
        max_length=args.utt_max_len,
        pad_value=pad_value,
        validation_split=args.validation_split,
        validation=True,
    )
    unpaired_train_dataset = dataloaders.MaskedTextDataset(
        text_file=args.text_file,
        char_map=args.char_map,
        default_alphabet=rttm_holder.letter2int,
        # Optional arguments start here
        mask_probability=args.mask_probability_text,
        force_mask_query=args.text_force_mask_query,
        force_unmask_query=args.text_force_unmask_query,
        #
        positive_samples_per_doc=args.positive_samples_per_doc_text,
        negative_samples_per_doc=args.negative_samples_per_doc_text,
        repeat=args.text_repetition,
        use_blank_token=args.text_use_blank_token,
        use_unk_token=args.text_use_unk_token,
        loss_weight=args.text_loss_weight,
        negative_samples_weight=args.text_negative_samples_weight,
        max_training_utt_len=args.text_max_training_utt_len,
        max_training_utt_len_word=args.text_max_training_utt_len_word,
        pad_value=pad_value,
    )

    if distributed:
        paired_train_dataloader = torch.utils.data.DataLoader(
            dataset=paired_train_dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Will be set for CompoundDataset below
            shuffle=False,  # conflicts with ElasticDistributedSampler
            pin_memory=False, # Will also be set for CompoundDataset below
            collate_fn=dataloaders.handle_batch,
            sampler=ElasticDistributedSampler(paired_train_dataset)
            
        )
        unpaired_train_dataloader = torch.utils.data.DataLoader(
            dataset=unpaired_train_dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Will be set for CompoundDataset below
            shuffle=False,  # conflicts with ElasticDistributedSampler
            pin_memory=False, # Will also be set for CompoundDataset below
            collate_fn=dataloaders.handle_batch,
            sampler=ElasticDistributedSampler(unpaired_train_dataset)
            
        )
        valid_dataloader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False, # Will also be set for CompoundDataset below
            collate_fn=dataloaders.handle_batch,
            sampler=ElasticDistributedSampler(validation_dataset)
            
        )
    else:
        paired_train_dataloader = torch.utils.data.DataLoader(
            dataset=paired_train_dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Will be set for CompoundDataset below
            shuffle=True,
            pin_memory=False, # Will also be set for CompoundDataset below
            collate_fn=dataloaders.handle_batch,
        )
        unpaired_train_dataloader = torch.utils.data.DataLoader(
            dataset=unpaired_train_dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Will be set for CompoundDataset below
            shuffle=True,
            pin_memory=False, # Will also be set for CompoundDataset below
            collate_fn=dataloaders.handle_batch,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=dataloaders.handle_batch,
        )

    dsets = {'speech': paired_train_dataloader, 'text': unpaired_train_dataloader}
    weights = {'speech': 1, 'text': args.text_sampling_weight}

    seed = 0 if distributed else None

    compound_dataset = dataloaders.CompoundDataset(dsets, weights, steps_per_epoch=args.steps_per_epoch,
                                                   pad_value=pad_value,
                                                   seed=seed,
                                                   )
    dataloader_extra_args = {}
    if args.num_workers > 0:
        dataloader_extra_args = {'prefetch_factor': 4}
    train_dataloader = dataloaders.DataLoader(compound_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=lambda y: y[0],
                                              **dataloader_extra_args)

    data_loaders = {'train': train_dataloader, 'test': valid_dataloader}

    document_model = models.MultitaskCompositeModel(args.document_encoder_json,
                                                    input_dims={1: featdim,
                                                                0: len(unpaired_train_dataset.alphabet)}, )

    # Extra dimension will be used for padding
    query_model = models.CompositeModel(args.query_encoder_json, input_dim=len(rttm_holder.letter2int) + 1)

    search_model = models.SimpleSearchNet(
        query_model,
        document_model,
        query_sum=args.use_query_sum,
        summarizer=args.query_sum_type,
        )
    trainer_conf = {}
    if args.trainer_json:
        with open(args.trainer_json) as _json:
            trainer_conf = json.load(_json)
    
    TrainerClass = training.DistributedDataParallelTrainingLoop if distributed else training.SimpleTrainingLoop

    if args.resume:
        training_loop = TrainerClass(outdir, device)
    else:
        training_loop = TrainerClass(
            search_model,
            device,
            pretrained_document_encoder=args.pretrained_document_encoder,
            pretrained_query_encoder=args.pretrained_query_encoder,
            **trainer_conf,
            )
    criterion = losses.MaskedXentMarginLoss(
        pad_value=pad_value,
        zero_weight=args.zero_weight,
        one_weight=args.one_weight,
        margin=args.loss_margin,
        )

    training_loop.train(outdir, criterion, data_loaders=data_loaders,
                        phases=['train', 'test'],
                        job_num_epochs=args.job_num_epochs,
                        )


if __name__ == '__main__':
    main()
