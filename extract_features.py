import os
import argparse
from typing import Dict

import numpy as np
import torch
from torch import nn
import torchaudio as ta
import transformers

import utils
import data_utils


def lengths_to_mask(lengths: torch.Tensor):
    if lengths is None:
        return None
    max_length = lengths.max()
    mask = torch.arange(max_length)[None, :].to(lengths.device) < lengths[:, None]
    return mask


class SimpleWavsDataset:  # (torch.utils.data.Dataset):
    '''
    Dataset object for loading data from a directory
    '''
    def __init__(self, datadir: str):

        # Get list of audio and form a dictionary mapping utterance ids to audio file paths
        audio_dir = datadir  # os.path.join(datadir, 'audio')
        list_of_audios = [os.path.join(audio_dir, a) for a in os.listdir(audio_dir) if a.endswith('.wav')]
        audio_dict = {os.path.basename(f).split('.')[0]: f for f in list_of_audios}
        self.audio_dict = audio_dict
        self.utterance_list = list(audio_dict.keys())

    def __getitem__(self, ind):
        # Get a single training example
        utterance_id = self.utterance_list[ind]
        audio_filename = self.audio_dict[utterance_id]

        # Load the audio
        speech_tensor, fs = ta.load(audio_filename)

        speech_features = speech_tensor.transpose(0, 1)

        return {'utterance_id': utterance_id,
                'speech_features': speech_features,
                'speech_length': len(speech_features),
                }


    def __len__(self):
        return len(self.utterance_list)

    def __iter__(self):
        for ind in range(len(self)):
            yield self[ind]


def make_batch(list_of_items):
    speech_features = [x['speech_features'] for x in list_of_items]
    speech_lengths = [x['speech_length'] for x in list_of_items]
    utterance_id = [x['utterance_id'] for x in list_of_items]

    speech_feature_batch = torch.zeros(len(speech_features), max(speech_lengths), speech_features[0].shape[-1])
    for i, (sp) in enumerate(speech_features):
        speech_feature_batch[i, :len(sp)] = sp

    return {'speech_features': speech_feature_batch,
            'speech_length': torch.tensor(speech_lengths),
            'utterance_id': utterance_id,
           }


class MelSpecExtractor(nn.Module):
    '''
    Module for extracting mel-spectrogram from audio. Optionally does SpecAugment on the fly
    '''
    def __init__(self, use_specaug=False):
        super().__init__()
        self.use_specaug = use_specaug
        self.spec = ta.transforms.Spectrogram(hop_length=160)
        self.mel_scale = ta.transforms.MelScale(
            n_mels=40,
            sample_rate=16000,
            )

    def forward(self, x: Dict[str, torch.Tensor]):
        speech = x['speech_features']
        speech_length = x['speech_length']

        speech_length = ((speech_length + self.spec.hop_length - 1)/ self.spec.hop_length).int()

        speech = speech.transpose(-1, -2)
        spectrogram = self.spec(speech)
        mel_spectrogram = self.mel_scale(spectrogram)[:, 0]
        x['output_speech_features'] = mel_spectrogram.transpose(-1, -2)
        x['output_speech_length'] = speech_length
        return x


class Wav2Vec2Wrapper(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m", layer_to_extract=16):
        super().__init__()
        self.encoder = transformers.Wav2Vec2Model.from_pretrained(model_name)
        self.strides = [_.conv.stride[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.kernel_sizes = [_.conv.kernel_size[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.layer_to_extract = layer_to_extract

    def forward(self,  x: Dict[str, torch.Tensor]):
        features = x['speech_features']
        speech_length = x['speech_length']

        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            speech_length = (1 + (speech_length - kernel_size) / stride).int()

        attention_mask = lengths_to_mask(speech_length).long()
        features = self.encoder(features.squeeze(-1), attention_mask=attention_mask, output_hidden_states=True)

        if self.layer_to_extract is None:
            x['output_speech_features'] = features.last_hidden_state
        else:
            x['output_speech_features'] = features.hidden_states[self.layer_to_extract]
        x['output_speech_length'] = speech_length
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--feature-type', '-f', choices=('mel', 'xlsr'), default='xlsr')
    parser.add_argument('audio_datadir')
    parser.add_argument('feats_outdir')

    args = parser.parse_args()

    dataset = SimpleWavsDataset(args.audio_datadir)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             collate_fn=make_batch,
                                             pin_memory=True,
                                             num_workers=4,
                                             )

    device = torch.device('cpu') if args.no_cuda else torch.device('cuda:0')

    if args.feature_type == 'xlsr':
        model = Wav2Vec2Wrapper()
    else:
        model = MelSpecExtractor()

    model = model.to(device)
    model.eval()

    feats_outdir = args.feats_outdir
    utils.chk_mkdir(feats_outdir)

    out_mats = []
    out_lengths = []
    out_keys = []

    # The feature matrices stored in RAM and saved to disk at the end
    # For larger datasets, consider using npy-append-array
    with torch.no_grad():
        for i, sample_batch in enumerate(dataloader):
            for k, v in sample_batch.items():
                if isinstance(v, torch.Tensor):
                    sample_batch[k] = v.to(device)
            out = model(sample_batch)
            out_mat = out['output_speech_features']
            out_length = out['output_speech_length']
            out_key = out['utterance_id']
            mats = [feat[:ln].cpu().numpy() for feat, ln in zip(out_mat, out_length)]

            out_mats += mats
            out_lengths.append(out_length.cpu().numpy())
            out_keys += out_key

    out_mats = np.concatenate(out_mats)
    out_offsets = np.concatenate(out_lengths).cumsum()[:-1]
    data_utils.save_mats(
        feats_outdir,
        filename='data.npy',
        mats=out_mats,
        offsets=out_offsets,
        keys=out_keys,
        )


if __name__ == '__main__':
    main()
