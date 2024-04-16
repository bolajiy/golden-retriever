import string
import random
import os
from typing import List, Tuple, Dict, Union, Iterable
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import data_utils


@dataclass
class Location:
    term: str
    document: str
    begin_time: float
    end_time: float
    score: float = 1


def remove_punctuation(string_x: str):
    return ''.join([_ for _ in string_x if _ not in string.punctuation])


def to_lower(string_x: str, tr: bool = False, special: Dict = None, sep='', use_spaces=False) -> str:
    if special is None:
        special = {}
    string_x = remove_punctuation(string_x)
    y = [x.lower() if x not in special.keys() else special[x]
         for x in string_x]
    if not use_spaces:
        y = [x for x in y if not(x.isspace())]
    return sep.join(y).strip()


def to_upper(string_x: str, tr: bool = False, special: Dict = None, use_spaces=False) -> str:
    if special is None:
        special = {}
    string_x = remove_punctuation(string_x)
    y = [x.upper() if x not in special.keys() else special[x]
         for x in string_x]
    if not use_spaces:
        y = [x for x in y if not(x.isspace())]
    return ''.join(y)


def read_rttm(rttm_file: str, word2phones_file: str = None, char_map: str = None, all_phones_override: dict = None,
              is_ctm=False) -> Tuple[Dict[str, List[Location]], Dict[str, None], dict, Dict[str, list]]:
    rttm_dict = {}
    word2int = {}
    all_words = {}
    all_spellings = {}
    all_phones = set()
    if char_map is not None and os.path.isfile(char_map):
        special = {line.strip().split(':')[0]: line.strip().split(':')[1]
                   for line in open(char_map, encoding='UTF-8')}
    else:
        special = {}
    if word2phones_file:
        word2phones = {to_lower(line.strip().split(';')[0]): line.strip().split(';')[1]
                       for line in open(word2phones_file, encoding='utf-8')}

        def word2phones_func(x):
            return word2phones[x].split()
    else:
        def word2phones_func(x):
            return [letter for letter in x]

    if is_ctm:
        locations = data_utils.read_ctm(rttm_file)
    else:
        locations = data_utils.read_rttm(rttm_file)
    for row, location in locations.iterrows():
        word = to_lower(location['word'].strip(), special=special).strip()
        spelling = word2phones_func(word)
        all_words[word] = None
        all_spellings[word] = spelling
        all_phones.update(spelling)
        location = Location(word, location.utterance, location.begin, location.end)
        if location.document not in rttm_dict:
            rttm_dict[location.document] = [location]
        else:
            rttm_dict[location.document].append(location)
    all_phones = {x: i for i, x in enumerate(all_phones)}
    if all_phones_override is not None:
        all_phones = all_phones_override

    for word in all_words:
        word2int[word] = [all_phones[phone] for phone in word2phones_func(word)]
    return rttm_dict, all_words, all_phones, word2int


def get_neighbors(list_of_locations: List[Location], neighborhood: int = 1, tolerance: float = 0.2) -> List[Location]:
    neighbors = [list_of_locations[i - neighborhood:i] for i in range(neighborhood, len(list_of_locations) + 1)]
    all_neighbors = []
    for candidate in neighbors:
        if not len(candidate):
            continue
        elif len(candidate) == 1:
            all_neighbors += candidate
        else:
            for i, w in enumerate(candidate[:-1]):
                if (candidate[i + 1].begin_time - w.end_time) > tolerance:
                    break
            else:
                term = " ".join([w.term for w in candidate])
                document = candidate[0].document
                begin_time = candidate[0].begin_time
                end_time = candidate[-1].end_time
                location = Location(term, document, begin_time, end_time)
                all_neighbors.append(location)
    return all_neighbors


class RttmHolder:
    def __init__(self, rttm_file: str, min_neighborhood: int = 1, max_neighborhood: int = 1, tolerance: float = 0.2,
                 min_kw_length: int = 1, max_kw_length: int = 100, allowed_docs=None, shuffle_list=True,
                 word2phones_file: str = None, char_map: str = None, is_ctm=False,
                 all_phones_override: dict = None,
                 ):
        if word2phones_file:
            self.word2phones = {to_lower(line.strip().split(';')[0]): line.strip().split(';')[1]
                                for line in open(word2phones_file, encoding='utf-8')}
        else:
            self.word2phones = None
        self.rttm_file = rttm_file
        self.min_neighborhood = min_neighborhood
        self.max_neighborhood = max_neighborhood
        self.min_kw_length = min_kw_length
        self.max_kw_length = max_kw_length
        self.tolerance = tolerance
        self.rttm_dict, self.all_words, self.letter2int, self.word2int = read_rttm(rttm_file,
                                                                                   word2phones_file=word2phones_file,
                                                                                   char_map=char_map,
                                                                                   is_ctm=is_ctm,
                                                                                   all_phones_override=all_phones_override,
                                                                                   )
        self.locations_dict = {}
        self.locations_dict_docs = {}
        self.locations_list = []
        for utt, doc in self.rttm_dict.items():
            if allowed_docs and utt not in allowed_docs:
                continue
            for n in range(min_neighborhood, max_neighborhood + 1):
                neighbors = sorted(get_neighbors(doc, neighborhood=n, tolerance=tolerance), key=lambda x: x.begin_time)
                neighbors = [x for x in neighbors
                             if self.min_kw_length <= len(''.join(x.term.split())) <= self.max_kw_length]
                if not neighbors:
                    continue
                try:
                    self.locations_dict[n] += neighbors
                except KeyError:
                    self.locations_dict[n] = [x for x in neighbors]
                try:
                    self.locations_dict_docs[utt] += neighbors
                except KeyError:
                    self.locations_dict_docs[utt] = [x for x in neighbors]
                self.locations_list += neighbors
        for utt, locs in self.locations_dict_docs.items():
            locs.sort(key=lambda x: x.begin_time)
        self.allowed_docs = [utt for utt in self.locations_dict_docs.keys()]
        if shuffle_list:
            random.shuffle(self.locations_list)
            random.shuffle(self.allowed_docs)

    def spell(self, words, word2phones=None):
        if word2phones is None:
            word2phones = self.word2phones
        spelling = []
        if word2phones is not None:
            for word in words.split():
                spelling += word2phones[word].split()
        else:
            for word in words.split():
                spelling += [self.letter2int[letter] for letter in word
                             if letter in self.letter2int]
        return spelling


class KwsTrainingDataset(Dataset):
    def __init__(self,
                 data: List[Tensor],
                 keys: Dict[str, int],
                 rttm_holder: RttmHolder,
                 negative_samples_per_doc: int = 3,
                 frame_length: float = 0.01,
                 max_length: float = 2.,
                 pad_value: int = -1000,
                 validation: bool = False,
                 validation_split: float = 0.1,
                 **kwargs):
        self.data = data
        self.data_dim = self.data[0].shape[-1]
        self.keys = keys
        self.rttm_holder = rttm_holder
        self.negative_samples_per_doc = negative_samples_per_doc
        self.frame_length = frame_length
        self.max_length = max_length
        self.max_length_frames = int(self.max_length / self.frame_length)
        self.pad_value = pad_value
        lens = [len(rttm_holder.spell(y.term)) for y in rttm_holder.locations_list]
        self.max_query_len = max(lens)
        self.query_dim = len(self.rttm_holder.letter2int)
        self.phone_mat = torch.eye(self.query_dim)
        self.is_exhaustive = False
        self.kwargs = kwargs

        validation_length = int(validation_split * len(self.rttm_holder.locations_list))
        if validation:
            self.allowed_locations = self.rttm_holder.locations_list[:validation_length]
        else:
            self.allowed_locations = self.rttm_holder.locations_list[validation_length:]
        self.allowed_locations = [x for x in self.allowed_locations
                                  if ((x.end_time - x.begin_time) // self.frame_length)
                                  < self.max_length_frames]

    def __len__(self) -> int:
        return len(self.allowed_locations)

    def get_sample(self, ind: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        positive_location = self.allowed_locations[ind]
        negative_locations = random.sample(self.allowed_locations, k=self.negative_samples_per_doc)
        positive_samples = [self.get_single_sample(positive_location, positive_location.document, positive=True)]
        negative_samples = [self.get_single_sample(positive_location, x.document, positive=True)
                            for x in negative_locations]
        all_samples = positive_samples + negative_samples
        query_batch, doc_batch, query_lens, document_lens, label_batch, query_mask, document_mask, sample_weights_batch = self.batchify(all_samples)
        query = {'features': query_batch, 'lengths': query_lens}
        document = {'features': doc_batch, 'lengths': document_lens}
        label = {'features': label_batch, 'lengths': document_lens, 'weights': sample_weights_batch}
        return {'query': query, 'document': document, 'label': label}

    def __getitem__(self, ind):
        return self.get_sample(ind)

    def get_single_sample(self, location: Location, document: str,
                          positive: bool = True, start_frame: int = None,
                          ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        split_term = location.term.split()
        term_int = sum([self.rttm_holder.word2int[x] for x in split_term], [])
        term = [x for x in term_int if x is not None]
        query = torch.tensor(term)
        data_id = self.keys[document]
        document_mat = self.data[data_id].clone()
        doc_len = document_mat.size(0)
        length = doc_len if doc_len <= self.max_length_frames else self.max_length_frames

        if positive:
            begin_frame = int(location.begin_time // self.frame_length)
            end_frame = int(location.end_time // self.frame_length)
        else:
            begin_frame = random.choice(range(0, doc_len - length + 1))
            end_frame = begin_frame + length

        dur = end_frame - begin_frame
        diff = length - dur
        d2 = diff // 2
        beg = begin_frame - d2
        end = end_frame + d2

        if beg <= 0:
            beg = 0
            end = beg + length
        elif end >= doc_len:
            end = min(doc_len, begin_frame + length)
            beg = end - length
        if start_frame is not None:
            beg = start_frame
            end = min(doc_len, beg + length)

        stamps = []
        labels = torch.zeros(doc_len)
        sample_weights = torch.ones(doc_len)
        for loc in self.rttm_holder.locations_dict_docs[document]:
            if loc.term == location.term:
                s0, s1 = int(loc.begin_time // self.frame_length), int(loc.end_time // self.frame_length)
                stamps.append((s0, s1))
                labels[stamps[-1][0]:stamps[-1][1]] = 1
                length = stamps[-1][1] - stamps[-1][0]
        doc = document_mat[beg:end]
        lab = labels[beg:end]
        sw = sample_weights[beg:end]
        return query, doc, lab, sw

    def batchify(self, all_samples: list) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        doc_batch = torch.zeros(len(all_samples), self.max_length_frames, self.data_dim)
        # Final embedding is used for padding
        query_batch = len(self.rttm_holder.letter2int) * torch.ones(len(all_samples), self.max_query_len).long()
        query_mask = torch.zeros(len(all_samples), self.max_query_len)
        label_batch = self.pad_value * torch.ones(len(all_samples), self.max_length_frames)
        document_mask = torch.zeros(len(all_samples), self.max_length_frames, self.data_dim)
        query_lens = torch.zeros(len(all_samples)).int()
        sample_weights_batch = torch.ones(len(all_samples), self.max_length_frames)
        document_lens = torch.zeros(len(all_samples), dtype=int)
        for i, (query, document, label, sample_weights) in enumerate(all_samples):
            query_batch[i, :len(query)] = query
            query_mask[i, :len(query)] = 1
            query_lens[i] = len(query)
            doc_batch[i, :len(document)] = document
            document_mask[i, :len(document)] = 1
            document_lens[i] = len(document)
            label_batch[i, :len(label)] = label
            sample_weights_batch[i, :len(sample_weights)] = sample_weights
        return query_batch, doc_batch, query_lens, document_lens, label_batch, query_mask, document_mask, sample_weights_batch


def handle_batch(batch_list):
    out_batch = {'query': {}, 'document': {}, 'label': {}}
    for key in out_batch.keys():
        batches = [batch[key] for batch in batch_list]
        inner_keys = batches[0].keys()
        out_batch[key] = {inner_key: torch.cat([batch[inner_key] for batch in batches]) for inner_key in inner_keys}
    return out_batch


class MaskedTextDataset(Dataset):
    def __init__(self, text_file, mask_probability=0.2, negative_samples_per_doc=3, repeat=1, pad_value=-1000,
                 positive_samples_per_doc=1,
                 loss_weight=1., has_uttid=True, min_ngram=1, max_ngram=3, min_kw_length=3, max_kw_length=100,
                 negative_samples_weight=1.,
                 char_map=None,
                 default_alphabet=None,
                 force_mask_query=False,
                 force_unmask_query=False,
                 use_spaces=False,
                 max_utt_len_allowed=None,
                 max_training_utt_len=None,
                 max_training_utt_len_word=None,
                 use_unk_token=False,
                 use_blank_token=False,
                 ):

        self.mask_probability = mask_probability
        self.force_mask_query = force_mask_query
        self.force_unmask_query = force_unmask_query
        assert not (force_mask_query and force_unmask_query), ("only one of force_mask_query and "
                                                               "force_unmask_query should be true")
        self.negative_samples_weight = negative_samples_weight
        self.loss_weight = loss_weight
        self.negative_samples_per_doc = negative_samples_per_doc
        self.positive_samples_per_doc = positive_samples_per_doc
        self.max_kw_length = max_kw_length
        self.pad_value = pad_value
        self.repeat = repeat
        self.max_training_utt_len = max_training_utt_len
        self.max_training_utt_len_word = max_training_utt_len_word

        if char_map is not None and os.path.isfile(char_map):
            special = {line.strip().split(':')[0]: line.strip().split(':')[1]
                       for line in open(char_map, encoding='UTF-8')}
        else:
            special = {}
        self.special = special
        self.use_spaces = use_spaces
        self.use_unk_token = use_unk_token
        self.use_blank_token = use_blank_token    
        self.sentences = {}

        self.locations_list = []
        self.locations_dict = {}
        self.locations_dict_docs = {}
        self.word_boundaries = {}
        self.max_utt_len = 0
        alphabet = set(' ') if self.use_spaces else set()
        all_lines = data_utils.read_txt(text_file, has_uttid)
        for row, line in all_lines.iterrows():
            if has_uttid:
                utt, sentence = line.utterance, line.text
            else:
                utt, sentence = str(row), line.text
            sentence = to_lower(sentence, special=special, use_spaces=True)
            if max_utt_len_allowed and len(sentence) > max_utt_len_allowed:
                continue

            sentence = ''.join([_ for _ in sentence if _ not in string.punctuation])
            alphabet = alphabet.union(sentence)
            self.sentences[utt] = sentence if self.use_spaces else ''.join(sentence.split())
            if len(self.sentences[utt]) > self.max_utt_len:
                self.max_utt_len = len(self.sentences[utt])
            doc = []
            sentence = sentence.split()
            begin = 0
            for word in sentence:
                end = begin + len(word)
                doc.append(Location(word, utt, float(begin), float(end)))
                begin = (end + 1) if self.use_spaces else end
            if 1 not in range(min_ngram, max_ngram + 1):
                n = 1
                neighbors = sorted(get_neighbors(doc, neighborhood=n, tolerance=1), key=lambda x: x.begin_time)
                neighbors = [x for x in neighbors
                             if min_kw_length <= len(''.join(x.term.split())) <= max_kw_length]
                try:
                    self.word_boundaries[utt] += neighbors
                except KeyError:
                    self.word_boundaries[utt] = [x for x in neighbors]

            for n in range(min_ngram, max_ngram + 1):
                neighbors = sorted(get_neighbors(doc, neighborhood=n, tolerance=1), key=lambda x: x.begin_time)
                neighbors = [x for x in neighbors
                             if min_kw_length <= len(''.join(x.term.split())) <= max_kw_length]
                if n == 1:
                    try:
                        self.word_boundaries[utt] += neighbors
                    except KeyError:
                        self.word_boundaries[utt] = [x for x in neighbors]
                if not neighbors:
                    continue
                try:
                    self.locations_dict[n] += neighbors
                except KeyError:
                    self.locations_dict[n] = [x for x in neighbors]
                try:
                    self.locations_dict_docs[utt] += neighbors
                except KeyError:
                    self.locations_dict_docs[utt] = [x for x in neighbors]
                self.locations_list += neighbors
        for utt in self.locations_dict_docs.keys():
            self.locations_dict_docs[utt].sort(key=lambda x: x.begin_time)
        if not self.use_spaces:
            alphabet = alphabet - {' '}
        self.alphabet = {k: v for v, k in enumerate(sorted(list(alphabet)))}
        if default_alphabet is not None:
            self.alphabet = default_alphabet
        if '<m>' not in self.alphabet.keys():
            self.alphabet['<m>'] = len(self.alphabet)
        if '<pad>' not in self.alphabet.keys():
            self.alphabet['<pad>'] = len(self.alphabet)
        if self.use_unk_token:
            self.alphabet['<unk>'] = len(self.alphabet)
        if self.use_blank_token:
            self.alphabet['<blank>']= len(self.alphabet)
        self.reverse_alphabet = {v: k for k, v in self.alphabet.items()}
        self.special_cache = {}

    def tokenize(self, phrase, alphabet=None, use_unk_token=None,):
        if alphabet is None:
            alphabet = self.alphabet
        if use_unk_token is None:
            use_unk_token = self.use_unk_token
        if not self.use_spaces:
            phrase = ' '.join(phrase.split())
        if use_unk_token:
            return [alphabet[k] if k in alphabet else alphabet['<unk>'] for k in phrase]
        else:
            return [alphabet[k] for k in phrase if k in alphabet]

    def detokenize(self, phrase_ten):
        return ''.join([self.reverse_alphabet[int(k)] for k in phrase_ten])

    def __len__(self):
        return len(self.locations_list)

    def nonconsecutive_mask(self, document_tensor, stamps=None, mask_token=None):
        if mask_token is None:
            mask_token = self.alphabet['<m>']
        mask = torch.rand_like(document_tensor.float()) >= self.mask_probability
        if self.force_mask_query and stamps:
            for start, end in stamps:
                mask[start:end] = 0
        elif self.force_unmask_query and stamps:
            for start, end in stamps:
                mask[start:end] = 1       
        document_tensor[mask == 0] = mask_token
        return document_tensor

    def get_single_sample(self, location: Location, document: str,
                          positive: bool = True, start_frame: int = None,
                          force_negative: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        def trim(doc):
            doc_len = len(doc)
            max_len_flag = self.max_training_utt_len or self.max_training_utt_len_word
            if max_len_flag and doc_len > max_len_flag:
                if self.max_training_utt_len:
                    length = self.max_training_utt_len
                else:
                    length = None
                if self.max_training_utt_len_word:
                    if length is None:
                        length = int(location.end_time - location.begin_time)
                    else:
                        length = min(length, int(location.end_time - location.begin_time))
                if positive:
                    begin_frame = int(location.begin_time)
                    end_frame = int(location.end_time)
                else:
                    begin_frame = random.choice(range(0, doc_len - length + 1))
                    end_frame = begin_frame + length
                # if start_frame is not None:
                #     begin_frame = start_frame

                dur = end_frame - begin_frame
                diff = length - dur
                d2 = diff // 2
                beg = begin_frame - d2
                end = end_frame + d2
                if beg <= 0:
                    beg = 0
                    end = beg + length
                elif end >= doc_len:
                    end = min(doc_len, begin_frame + length)
                    beg = end - length
                return beg, end
            else:
                return ()

        document_tensor = torch.tensor(self.tokenize(self.sentences[document]))
        query_tensor = torch.tensor(self.tokenize(location.term))
        labels = torch.zeros_like(document_tensor)
        sample_weights = self.negative_samples_weight * self.loss_weight * torch.ones_like(document_tensor).float()

        stamps = []
        for loc in self.locations_dict_docs[document]:
            if loc.term == location.term:
                s0, s1 = int(loc.begin_time), int(loc.end_time)
                stamps.append((s0, s1))
                labels[stamps[-1][0]:stamps[-1][1]] = 1
                sample_weights[stamps[-1][0]:stamps[-1][1]] = self.loss_weight
        if self.use_unk_token:
            sample_weights[document_tensor == self.alphabet['<unk>']] = 0

        document_tensor = self.nonconsecutive_mask(document_tensor, stamps)

        if self.use_blank_token:
            document_tensor = document_tensor.unsqueeze(-1).repeat_interleave(self.repeat, dim=-1)
            document_tensor[:, 1:] = self.alphabet['<blank>']
            document_tensor = document_tensor.view(-1)
        else:
            document_tensor = document_tensor.long().repeat_interleave(self.repeat, dim=-1)
        labels = labels.repeat_interleave(self.repeat, dim=-1)
        sample_weights = sample_weights.repeat_interleave(self.repeat, dim=-1)
        bounds = trim(document_tensor)
        if bounds:
            b, e = bounds
            document_tensor = document_tensor[b:e]
            labels = labels[b:e]
            sample_weights = sample_weights[b:e]
        return query_tensor, document_tensor, labels, sample_weights

    def get_sample(self, ind, getlen=False):
        if getlen:
            return len(self)
        positive_location = self.locations_list[ind]
        negative_locations = random.sample(self.locations_list, k=self.negative_samples_per_doc)
        positive_samples = [self.get_single_sample(positive_location, positive_location.document, positive=True)
                            for _ in range(self.positive_samples_per_doc)]
        negative_samples = [self.get_single_sample(positive_location, x.document, positive=True)
                            for x in negative_locations]
        all_samples = positive_samples + negative_samples
        return self.batchify(all_samples)

    def __getitem__(self, ind):
        return self.get_sample(ind)

    def batchify(self, all_samples: list) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        doc_batch = self.alphabet['<pad>'] * torch.ones(len(all_samples), self.max_utt_len * self.repeat).long()
        # Final embedding is used for padding
        query_batch = self.alphabet['<pad>'] * torch.ones(len(all_samples), self.max_kw_length).long()
        # query_batch = len(self.alphabet) * torch.ones(len(all_samples), self.max_kw_length).long()
        query_mask = torch.zeros(len(all_samples), self.max_kw_length)
        label_batch = self.pad_value * torch.ones(len(all_samples), self.max_utt_len * self.repeat)
        document_mask = torch.zeros(len(all_samples), self.max_utt_len * self.repeat).long()
        query_lens = torch.zeros(len(all_samples)).int()
        doc_lens = torch.zeros(len(all_samples)).int()
        sample_weights_batch = torch.ones(len(all_samples), self.max_utt_len * self.repeat)
        for i, (query, document, label, sample_weights) in enumerate(all_samples):
            query_batch[i, :len(query)] = query
            query_mask[i, :len(query)] = 1
            query_lens[i] = len(query)
            doc_lens[i] = len(document)
            doc_batch[i, :len(document)] = document
            document_mask[i, :len(document)] = 1
            label_batch[i, :len(label)] = label
            sample_weights_batch[i, :len(sample_weights)] = sample_weights
        query = {'features': query_batch, 'lengths': query_lens}
        document = {'features': doc_batch, 'lengths': doc_lens}
        label = {'features': label_batch, 'lengths': doc_lens, 'weights': sample_weights_batch}
        return {'query': query, 'document': document, 'label': label,}



class CompoundDataset(Dataset):
    def __init__(self, dataloaders: Dict[str, DataLoader],
                 weights: Dict[str, float] = None,
                 steps_per_epoch: int = 100000,
                 deterministic: bool = False,
                 pad_value: int = -1000,
                 seed: int = None,
                 labels: Dict[str, str] = None,
                 extra_labels: Dict[str, Dict[str, str]] = None,
                 return_only_batch: bool = False,
                 ):
        self._dataloaders = dataloaders
        self.dataloaders = {k: iter(v) for k, v in dataloaders.items()}
        if weights is None:
            weights = {k: 1. for k in self.dataloaders.keys()}
        weights = {k: int(10000 * v) for k, v in weights.items()}
        self.weights = weights
        if steps_per_epoch is None:
            steps_per_epoch = sum([len(d) for d in self.dataloaders.values()])
        self.steps_per_epoch = steps_per_epoch
        self.deterministic = deterministic
        self.pad_value = pad_value
        if seed is None:
            self.dataset_sequence = None
        else:
            # When running with multiple GPU workers, this block ensures that all workers access the same dataset at a
            # time. Otherwise, the all_reduce would fail as some parameters could end up with no gradients from some workers
            data_utils.np.random.seed(seed)
            _p = data_utils.np.array([weights[k] for k in self.dataloaders.keys()])
            _p = _p/_p.sum()
            self.dataset_sequence = data_utils.np.random.choice(len(self._dataloaders), size=1000000,
                                                                    p=_p)
        self.labels = labels
        self.extra_labels = extra_labels
        self.return_only_batch = return_only_batch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, item):
        datasets = list(self.weights.keys())
        weights = [self.weights[k] for k in datasets]
        if self.dataset_sequence is None:
            dataset = random.sample(datasets, 1, counts=weights)[0]
        else:
            dataset_ind = self.dataset_sequence[item % len(self.dataset_sequence)]
            # dataset_ind = item % len(datasets)  # Warning: This makes things uniform and deterministic
            dataset = datasets[dataset_ind]
        try:
            batch = next(self.dataloaders[dataset])
        except StopIteration:
            self.dataloaders[dataset] = iter(self._dataloaders[dataset])
            batch = next(self.dataloaders[dataset])
        if self.return_only_batch:
            return batch
        batch['dataset'] = dataset
        if self.labels is not None:
            batch['dataset'] = self.labels[dataset]
        if self.extra_labels is not None:
            for label_name, labels in self.extra_labels.items():
                batch[label_name] = labels[dataset]
        return batch


_dataset_names = {'KwsTrainingDataset': KwsTrainingDataset,
                  }

def get_dataset(dataset_name):
    return _dataset_names[dataset_name]
