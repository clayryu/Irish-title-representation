from collections import defaultdict
from pathlib import Path
import random
import yaml
from omegaconf import OmegaConf
from pyabc import pyabc
# from str_utils import split_note
import pandas as pd
import numpy as np
import json
import time
import pickle
import copy

import torch
from torch.nn.utils.rnn import pack_sequence
from sentence_transformers import SentenceTransformer

# from vocab_utils import TokenVocab, MusicTokenVocab
from pre_str_utils import split_note
import pre_vocab_utils

def convert_token(token):
  if isinstance(token, pyabc.Note):
    return f"{token.midi_pitch}//{token.duration}"
  if isinstance(token, pyabc.Rest):
    return f"{0}//{token.duration}"
  text = token._text
  if '"' in text:
    text = text.replace('"', '')
  if text == 'M':
    return None
  if text == '\n':
    return None
  if text in ['u', 'v', '.']:
    return None
  return text

def is_used_token(token):
  if isinstance(token, pyabc.ChordSymbol):
    return False
  return True

def is_valid_tune(tune):
  header = tune.header
  if 'key' not in header:
    return False
  if 'meter' not in header:
    return False

  for token in tune.tokens:
    if isinstance(token, pyabc.BodyField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    if isinstance(token, pyabc.InlineField):
      '''
      중간에 key나 meter가 바뀌는 경우 사용하지 않음
      '''
      return False
    token_text = convert_token(token)
    if token_text == '|:1':
      return False
    if token_text == ':||:4':
      return False
    # TODO: 파트가 여러개인 경우 처리하는 부분이 필요함
  # for i in range(1,10):
  #   last_token = tune.tokens[-i]
    
  #   if '|' in  last_token._text:
  #     return True
  #   elif isinstance(last_token, pyabc.Note):
  #     return False
  return True


def title_to_list_of_str(tune):
    return [vocab for vocab in tune.title.split(' ')]

def read_abc(file_path):
    with open(file_path, 'r') as f:
        fr = f.read()
    tune = pyabc.Tune(abc=fr)
    return tune

def read_tunes(file_path):
    with open(file_path, 'r') as f:
        fr = f.read()
    tunes = pyabc.Tunes(abc=fr)
    return tunes

def prepare_abc(paths: list):
  tune_list = []
  error_list = []
  for path in paths:
    try:
      # tune = read_abc(path)
      # tune_list.append(tune)
      tunes = read_tunes(path)
      if len(tunes.tunes) == 0:
        error_list.append(path)
      else:
        tune_list += tunes.tunes
    except:
      error_list.append(path)
  return tune_list, error_list

def decode_melody(melody, vocab):
    '''
    melody (torch.Tensor): model's prediction. Assume that melody has shape of [T]
    '''
    list_of_string = [vocab[token] for token in melody.tolist()[1:]]
    abc_decoded = ''.join(list_of_string)
    return abc_decoded

def read_yaml(yml_path):
  with open(yml_path, 'r') as f:
    yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
  config = OmegaConf.create(yaml_obj)
  return config

def get_emb_total_size(config, vocab):
  vocab_size_dict = vocab.get_size()

  emb_param = config.emb
  total_size = 0 
  for key in vocab_size_dict.keys():
    size = int(emb_param[key] * emb_param.emb_size)
    total_size += size
    emb_param[key] = size
  emb_param.total_size = total_size
  config.emb = emb_param
  return config

def update_config(config, args):
  config = OmegaConf.merge(config, args)
  return config

class ABCset:
  def __init__(self, dir_path, vocab_path=None, num_limit=None, vocab_name='dict', config=None, pre_vocab=None):
    self.config = config
    if isinstance(dir_path, str) or isinstance(dir_path, Path):
      # self.tune_length = config.dataset_tune_length
      # self.pretrnd_ttl_emb_type = config.pretrnd_ttl_emb_type
      self.dir = Path(dir_path)
      self.abc_list = list(self.dir.rglob('*.abc')) + list(self.dir.rglob('*.ABC'))
      self.abc_list.sort()
      if num_limit is not None:
        self.abc_list = self.abc_list[:num_limit]
      self.tune_list, error_list = prepare_abc(self.abc_list)
    elif isinstance(dir_path, list):
      if isinstance(dir_path[0], pyabc.Tune):
        print("Handling dataset input as a tune list")
        self.tune_list = dir_path
      elif isinstance(dir_path[0], Path):
        print("Handling dataset input as a abc path list")
        self.abc_list = dir_path
        self.tune_list, error_list = prepare_abc(self.abc_list)
    else:
      print(f'Error: Unknown input type: {type(dir_path)}')
    self.tune_list = [tune for tune in self.tune_list if self._check_tune_validity(tune)]
    if self.config.data_params.make_vocab:
      self._get_vocab(vocab_path, vocab_name)
    else:
      self.vocab = pre_vocab
    self._prepare_data()
    self.augmentor = Augmentor(config.data_params.key_aug, self.tune_list)

  def _check_tune_validity(self, tune):
    return True # NotImplemented

  def _prepare_data(self):
    self.data = [self._tune_to_list_of_str(tune) for tune in self.tune_list]
    self.header = [tune.header for tune in self.tune_list]

  def _get_vocab(self, vocab_path):
    entire_char_list = [token for tune in self.data for token in tune]
    self.vocab = ['<pad>', '<start>', '<end>'] + sorted(list(set(entire_char_list)))
    self.tok2idx = {key: i for i, key in enumerate(self.vocab)}
    #self.idx2tok = self.vocab

  def _get_unique_header_tokens(self):
    target_keys = ['key', 'meter', 'unit note length', 'rhythm']
    head_for_keys = ['K: ', 'M: ', 'L: ', 'R: ']
    output = []
    for head in self.header:
      for i, key in enumerate(target_keys):
        if key in head:
          output.append(head_for_keys[i] + head[key])
    return sorted(list(set(output)))

  def _tune_to_list_of_str(self, tune):
    return [convert_token(token) for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    tune_in_idx = [self.tok2idx[token] for token in tune]
    
    tune_tensor = torch.LongTensor(tune_in_idx)
    
    return tune_tensor[:-1], tune_tensor[1:]

  def get_trainset(self):
    return ABCset(self.tune_list)

def pack_collate(raw_batch:list):
    '''
  This function takes a list of data, and returns two PackedSequences
  
  Argument
    raw_batch: A list of MelodyDataset[idx]. Each item in the list is a tuple of (melody, shifted_melody)
               melody and shifted_melody has a shape of [num_notes (+1 if you don't consider "start" and "end" token as note), 2]
  Returns
    packed_melody (torch.nn.utils.rnn.PackedSequence)
    packed_shifted_melody (torch.nn.utils.rnn.PackedSequence)

  TODO: Complete this function
    '''  
    
    melody = [mel_pair[0] for mel_pair in raw_batch]
    shifted_melody = [mel_pair[1] for mel_pair in raw_batch]
    
    packed_melody = pack_sequence(melody, enforce_sorted=False)
    packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
    
    if len(raw_batch[0]) == 2:
      return packed_melody, packed_shifted_melody
    elif len(raw_batch[0]) == 3:
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
      packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
      return packed_melody, packed_shifted_melody, packed_measure_numbers
    else:
      raise ValueError("Unknown raw_batch format")



class PitchDurSplitSet(ABCset):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, vocab_name='TokenVocab', config=None, pre_vocab=None):
    super().__init__(dir_path, vocab_path, num_limit, vocab_name, config, pre_vocab)

  def _get_vocab(self, vocab_path, vocab_name):
    entire_char_list = [splitted for tune in self.data for token in tune for splitted in split_note(token)]
    unique_header_list = self._get_unique_header_tokens()
    unique_char_list = sorted(list(set(entire_char_list)))  + unique_header_list
    # self.vocab = TokenVocab(vocab_path, unique_char_list)
    self.vocab = getattr(pre_vocab_utils, vocab_name)(vocab_path, unique_char_list)

  def __getitem__(self, idx):
    tune = ['<start>'] + self.data[idx] + ['<end>']
    header = self.header[idx]
    tune_in_idx = [self.vocab(token, header) for token in tune]
    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(header))

    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    return tune_tensor[:-1], tune_tensor[1:]


class MeasureOffsetSet(PitchDurSplitSet):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, vocab_name='TokenVocab', config=None, pre_vocab=None):
    super().__init__(dir_path, vocab_path, num_limit, vocab_name, config, pre_vocab)

  def _check_tune_validity(self, tune):
    if len(tune.measures) == 0 or tune.measures[-1].number == 0:
      return False
    if not tune.is_ending_with_bar:
      return False
    if tune.is_tune_with_full_measures or tune.is_incomplete_only_one_measure:
      if is_valid_tune(tune):
        return True
    else:
      return False

  def _prepare_data(self):
    data = [ [self._tune_to_list_of_str(tune), tune.header] for tune in self.tune_list]
    self.data = [x[0] for x in data]
    self.header = [x[1] for x in data]
    # self.header = [tune.header for tune in self.tune_list if tune.is_tune_with_full_measures ]

  def get_str_m_offset(self, token):
    if token.measure_offset is not None:
      return 'm_offset:'+str(float(token.measure_offset))
    else:
      return 'm_offset:'+str(token.measure_offset)

  def _tune_to_list_of_str(self, tune):

    converted_tokens = [ [i, convert_token(token)] for i, token in enumerate(tune.tokens) if is_used_token(token)]
    converted_tokens = [token for token in converted_tokens if token[1] is not None]

    measure_infos = [ ['m_idx:'+str(tune.tokens[i].meas_offset_from_repeat_start),  self.get_str_m_offset(tune.tokens[i])]  for i,_ in converted_tokens]
    last_duration = tune.tokens[converted_tokens[-1][0]].duration if hasattr(tune.tokens[converted_tokens[-1][0]], 'duration') else 0
    last_offset = tune.tokens[converted_tokens[-1][0]].measure_offset + last_duration
    measure_infos += [[measure_infos[-1][0], f'm_offset:{last_offset}']] 
    converted_tokens_w_start = ['<start>'] + [token[1] for token in converted_tokens]

    combined = [ [tk] + meas for tk, meas in zip(converted_tokens_w_start, measure_infos)]

    return combined
    # [[token, 'm_idx:'+str(tune.tokens[i+1].meas_offset_from_repeat_start), 'm_offset:'+str(tune.tokens[i].measure_offset)  ]for i, token in enumerate(converted_tokens)]
    # return [ [convert_token(token), 'm_idx:'+str(token.meas_offset_from_repeat_start), 'm_offset:'+str(token.measure_offset)] 
    #           for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def _get_measure_info_tokens(self):
    return sorted(list(set([info for tune in self.data for token in tune for info in token[1:]])))

  def _get_vocab(self, vocab_path, vocab_name):
    entire_char_list = [splitted for tune in self.data for token in tune for splitted in split_note(token[0])]
    unique_header_list = self._get_unique_header_tokens()
    unique_measure_info_list = self._get_measure_info_tokens()
    unique_char_list = sorted(list(set(entire_char_list)))  + unique_header_list + unique_measure_info_list
    # self.vocab = TokenVocab(vocab_path, unique_char_list)
    # self.vocab = MusicTokenVocab(vocab_path, unique_char_list)
    self.vocab = getattr(pre_vocab_utils, vocab_name)(vocab_path, unique_char_list)

  def __getitem__(self, idx):
    tune = self.data[idx] + [['<end>', '<end>', '<end>']]
    header = self.header[idx]
    tune_in_idx = [self.vocab(token, header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    assert tune_tensor.shape[-1] == 4
    header_tensor = torch.LongTensor(self.vocab.encode_header(header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    return tune_tensor[:-1], tune_tensor[1:]

class MeasureNumberSet(MeasureOffsetSet):
  def __init__(self, dir_path, vocab_path=None, num_limit=None, vocab_name='MusicTokenVocab', config=None, pre_vocab=None):
    super().__init__(dir_path, vocab_path, num_limit, vocab_name, config, pre_vocab)

  def _get_measure_info_tokens(self):
    return sorted(list(set([info for tune in self.data for token in tune for info in token[1:-1]])))

  def _tune_to_list_of_str(self, tune):
    converted_tokens = [ [i, convert_token(token)] for i, token in enumerate(tune.tokens) if is_used_token(token)]
    converted_tokens = [token for token in converted_tokens if token[1] is not None]

    measure_infos = [ ['m_idx:'+str(tune.tokens[i].meas_offset_from_repeat_start), self.get_str_m_offset(tune.tokens[i]), tune.tokens[i].measure_number]  for i,_ in converted_tokens]

    assert '|' in converted_tokens[-1][1], f"Last token should be barline, {converted_tokens[-1]}"
    # last_duration = tune.tokens[converted_tokens[-1][0]].duration if hasattr(tune.tokens[converted_tokens[-1][0]], 'duration') else 0
    # last_offset = tune.tokens[converted_tokens[-1][0]].measure_offset + last_duration
    # measure_infos += [[measure_infos[-1][0], f'm_offset:{last_offset}', measure_infos[-1][2]]] 
    measure_infos += [[f'm_idx:{str(tune.tokens[converted_tokens[-1][0]].meas_offset_from_repeat_start+1)}', 
                       'm_offset:0.0', 
                       measure_infos[-1][2]+1]]
    converted_tokens_w_start = ['<start>'] + [token[1] for token in converted_tokens]

    combined = [ [tk] + meas for tk, meas in zip(converted_tokens_w_start, measure_infos)]

    return combined
    return [ [convert_token(token), 'm_idx:'+str(token.meas_offset_from_repeat_start), 'm_offset:'+str(token.measure_offset), token.measure_number] 
              for token in tune.tokens if is_used_token(token) and convert_token(token) is not None]

  def filter_tune_by_vocab_exists(self):
    new_tunes = []
    new_headers = []
    for tune, header  in zip(self.data, self.header):
      converted_tune = [x[:-1] for x in tune]
      try:
        [self.vocab(token, header) for token in converted_tune]
        new_tunes.append(tune)
        new_headers.append(header)
        # print('tune added')
      except Exception as e:
        # print(e)
        continue
    self.data = new_tunes
    self.header = new_headers

  def filter_token_by_vocab_exists(self):
    new_tunes = []
    new_headers = []
    for tune, header  in zip(self.data, self.header):
      filtered_tune = []
      converted_tune = [x[:-1] for x in tune]
      for token in converted_tune:
        try:
          self.vocab(token, header)
          filtered_tune.append(token)
        except Exception as e:
          continue
      if len(filtered_tune)>0:
        new_tunes.append(filtered_tune)
        new_headers.append(header)
        print('tune added')
    self.data = new_tunes
    self.header = new_headers

  def __getitem__(self, idx):
    # tune = [['<start>','<start>','<start>' ]] + [x[:-1] for x in self.data[idx]] + [['<end>', '<end>', '<end>']]
    tune = [x[:-1] for x in self.data[idx]]  + [['<end>', '<end>', '<end>']]

    '''
    <start> A A B B | C
            0 1 2 3 4 0 
            0 0 0 0 0 1
    '''
    measure_numbers = [x[-1] for x in self.data[idx]]
    header = self.header[idx]
    tune, new_key = self.augmentor(tune, header)
    new_header = header.copy()
    new_header['key'] = new_key

    tune_in_idx = [self.vocab(token, new_header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(new_header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)
    # if sum([a>=b for a, b in zip(torch.max(tune_tensor, dim=0).values.tolist(), [x for x in self.vocab.get_size().values()])]) != 0:
    #   print (tune_tensor)

    return tune_tensor[:-1], tune_tensor[1:], torch.tensor(measure_numbers, dtype=torch.long)

  def get_trainset(self, ratio=20):
    train_abc_list = [x for x in self.abc_list if not x.stem.isdigit() or int(x.stem) % ratio != 0]
    trainset =  MeasureNumberSet(train_abc_list, None, make_vocab=False, key_aug=self.augmentor.aug_type)
    trainset.vocab = self.vocab
    return trainset

  def get_testset(self, ratio=20):
    test_abc_list = [x for x in self.abc_list if 'the_session' in x.parent.name and int(x.stem) % ratio == 0]
    if len(test_abc_list) == 0:
      test_abc_list = [x for x in self.abc_list if int(x.stem) % 10 == 0]
    testset =  MeasureNumberSet(test_abc_list, None, make_vocab=False, key_aug=None)
    testset.vocab = self.vocab
    return testset

# using tune length to filter tune length inside the dataset not in pack collate
# only work with dataset which have one tune in each title
class ABCsetTitle_tunelength(MeasureNumberSet):
  def _init__(self, dir_path, vocab_path=None, num_limit=None, vocab_name='MusicTokenVocab', config=None, pre_vocab=None):
    super().__init__(dir_path, vocab_path, num_limit, vocab_name, config, pre_vocab)
    #self._get_title_emb()
  
  def _prepare_data(self):
    data = [ [self._tune_to_list_of_str(tune), tune.header, tune.header["tune title"]] for tune in self.tune_list]
    
    # sampling sequence in dataset
    self.data=[]
    self.header = []
    self.title_in_text_avail = []
    if self.tune_length is not None:
      for x in data:
        if len(x[0]) < self.tune_length:
          continue
        sampled_num = random.randint(0,len(x[0])-self.tune_length)
        # self.data.append(x[0][sampled_num:sampled_num+self.tune_length])
        self.data.append(x[0][0:self.tune_length])
        self.header.append(x[1])
        self.title_in_text_avail.append(x[2])
    else:
      self.data = [x[0] for x in data]
      self.header = [x[1] for x in data]
      self.title_in_text_avail = [x[2] for x in data]
    #def _get_title_emb(self):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    self.ttl2emb = model.encode(self.title_in_text_avail, device='cuda')
  
  def __len__(self):
    return len(self.data)
      
  def __getitem__(self, idx):
    # tune = [['<start>','<start>','<start>' ]] + [x[:-1] for x in self.data[idx]] + [['<end>', '<end>', '<end>']]
    tune = [x[:-1] for x in self.data[idx]]  + [['<end>', '<end>', '<end>']]

    '''
    <start> A A B B | C
            0 1 2 3 4 0 
            0 0 0 0 0 1
    '''
    measure_numbers = [x[-1] for x in self.data[idx]]
    header = self.header[idx]
    tune, new_key = self.augmentor(tune, header)
    new_header = header.copy()
    new_header['key'] = new_key

    tune_in_idx = [self.vocab(token, new_header) for token in tune]

    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(new_header))
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)
    # if sum([a>=b for a, b in zip(torch.max(tune_tensor, dim=0).values.tolist(), [x for x in self.vocab.get_size().values()])]) != 0:
    #   print (tune_tensor)
    
    title = self.ttl2emb[idx]
    title_tensor = torch.FloatTensor(title)

    return tune_tensor[:-1], title_tensor, torch.tensor(measure_numbers, dtype=torch.long)

# using variation abc of tune. not constrain the length of tune, let it processed in pack collate
# work with every dataset with variation abc in each title
class ABCsetTitle_vartune(MeasureNumberSet):
  def _init__(self, dir_path, vocab_path=None, num_limit=None, make_vocab=False, key_aug=None, vocab_name='MusicTokenVocab', config=None, pre_vocab=None):
    super().__init__(dir_path, vocab_path, num_limit, vocab_name, config, pre_vocab)
    #self._get_title_emb()
  
  def _str_to_tensor(self, tune_str, tune_header):
    return [self.vocab(token, tune_header) for token in tune_str]

  def _prepare_data(self):
    start_time = time.time()
    self.abc_tensor = defaultdict(list)
    self.header = defaultdict(list)
    title_to_del = ['Untitled', 'x']
    # if self.config.language_detect is not None:
    #   with open(self.config.language_detect, "rb") as f:
    #     title_to_del += pickle.load(f)

    # to lower the time consumed in batch especially in the tune_in_idx
    # let's calculate the tune_in_idx in advance for every tune
    # process is simple : token(str) to idx to tensor
    for tune in self.tune_list:
      if tune.header["tune title"] not in title_to_del:
        tune_str = self._tune_to_list_of_str(tune)
        tune_str = [x[:-1] for x in tune_str] + [['<end>', '<end>', '<end>']]
        self.abc_tensor[tune.header["tune title"]].append(self._str_to_tensor(tune_str, tune.header))
        self.header[tune.header["tune title"]].append(tune.header)
    self.idx2ttl = list(self.abc_tensor.keys())
    
    df_embedding = pd.read_csv(self.config.text_embd_params.embd_csv_path)
    df_embedding[self.config.text_embd_params.model_type] = df_embedding[self.config.text_embd_params.model_type].apply(lambda x: np.array(eval(x)))
    self.dict_embedding = df_embedding.set_index('title')[self.config.text_embd_params.model_type].to_dict()

    print (f"Time taken to prepare data : {time.time() - start_time}")

  def __len__(self):
    return len(self.abc_tensor)
      
  def __getitem__(self, idx):
    picked_ttl = self.idx2ttl[idx]
    sampled_idx = random.randint(0, len(self.abc_tensor[picked_ttl])-1)
    # example of x : ['<start>', 'm_idx:0', 'm_offset:0.0', 0] so x[:-1] = ['<start>', 'm_idx:0', 'm_offset:0.0']
    # tune = [x[:-1] for x in self.data[picked_ttl][sampled_idx]]  + [['<end>', '<end>', '<end>']] # ['<start>', 'm_idx:0', 'm_offset:0.0', 8]
    header = self.header[picked_ttl][sampled_idx]

    measure_numbers = [x[-1] for x in self.abc_tensor[picked_ttl][sampled_idx]]

    # if not using augmentation code out for lower calculation cost
    # tune, new_key = self.augmentor(tune, header)
    # new_header = header.copy()
    # new_header['key'] = new_key
    new_header = header

    #tune_in_idx = [self.vocab(token, new_header) for token in tune] # the original code
    #tune_in_idx = [self.vocab_dictionary(token[0]) + self.vocab_dictionary.encode_m_idx(token[1]) + self.vocab.encode_m_offset(token[2], new_header) for token in tune]
    # this is the new code to calculate the tune_in_idx in dictionary
    # but hash table is not good for this case, because the tokens of every tune is too large
    tune_in_idx = self.abc_tensor[picked_ttl][sampled_idx]

    tune_tensor = torch.LongTensor(tune_in_idx)
    header_tensor = torch.LongTensor(self.vocab.encode_header(new_header))
    # train rnn model without genre information
    if self.config.general.input_feat == 'all_except_genre':
      header_tensor[3] = 0
    elif self.config.general.input_feat == 'melody_only':
      header_tensor = torch.zeros_like(header_tensor)
    tune_tensor = torch.cat([tune_tensor, header_tensor.repeat(len(tune_tensor), 1)], dim=-1)

    #title = self.ttl2emb[idx]
    if picked_ttl.split(' ')[-1] == 'The':
      list_title = picked_ttl.split(' ')[:-1]
      list_title.insert(0, 'The')
      picked_ttl = ' '.join(list_title)[:-1]
    title_tensor = self.dict_embedding[picked_ttl]
    title_tensor = torch.FloatTensor(title_tensor)
    return tune_tensor[:-1], title_tensor, torch.tensor(measure_numbers, dtype=torch.long), picked_ttl

  def parse_meter(self, meter):
    # example: meter = 4/4
    numer, denom = meter.split('/')
    is_compound = int(numer) in [6, 9, 12]
    is_triple = int(numer) in [3, 9]
    return numer, denom, is_compound, is_triple

def get_tunes_from_abc_fns(abc_fns):
  tunes = []
  errors = []
  for fn in abc_fns:
    try:
      with open(fn, 'r') as f:
        abc = f.read()
      tunes.append(pyabc.Tunes(abc).tunes)
    except:
      errors.append(fn)
  return tunes, errors

class Dataset_4feat_title():
  def __init__(self, title_list, genre_list, key_list, meter_list, unit_note_length_list, num_limit=None):
    if num_limit is not None:
      self.title_list = title_list[:num_limit]
      self.genre_list = genre_list[:num_limit]
      self.key_list = key_list[:num_limit]
      self.meter_list = meter_list[:num_limit]
      self.unit_note_length_list = unit_note_length_list[:num_limit]
    else:
      self.title_list = title_list
      self.genre_list = genre_list
      self.key_list = key_list
      self.meter_list = meter_list
      self.unit_note_length_list = unit_note_length_list

    self.genre2idx = {genre:idx for idx, genre in enumerate(set(self.genre_list))}
    self.key2idx = {key:idx for idx, key in enumerate(set(self.key_list))}
    self.meter2idx = {meter:idx for idx, meter in enumerate(set(self.meter_list))}
    self.unit_note_length2idx = {unit_note_length:idx for idx, unit_note_length in enumerate(set(self.unit_note_length_list))}

    self.__prepare_data()

  def __prepare_data(self):
    df_embedding = pd.read_csv('unq_ttl_emb_6283_melody.csv')
    df_embedding["ST_titleonly_6283"] = df_embedding["ST_titleonly_6283"].apply(lambda x: np.array(eval(x)))
    self.dict_embedding = df_embedding.set_index('title')["ST_titleonly_6283"].to_dict()

  def get_vocab(self):
    return self.genre2idx, self.key2idx, self.meter2idx, self.unit_note_length2idx

  def __len__(self):
    return len(self.title_list)
  
  def __getitem__(self, idx):
    title = self.title_list[idx]
    title_tensor = torch.FloatTensor(self.dict_embedding[title])

    genre = self.genre_list[idx]
    key = self.key_list[idx]
    meter = self.meter_list[idx]
    unit_note_length = self.unit_note_length_list[idx]

    header_tensor = torch.FloatTensor([self.genre2idx[genre], self.key2idx[key], self.meter2idx[meter], self.unit_note_length2idx[unit_note_length]])

    return header_tensor, title_tensor, torch.tensor([0]), title

class Augmentor:
  def __init__(self, aug_type, tune_list):
    self.aug_type = aug_type

    # self.chromatic = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    self.chromatic = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    self.note2idx = {note: i for i, note in enumerate(self.chromatic)}

    self.key_stats = self.get_key_stats(tune_list)

  def get_key_stats(self, tune_list):
    counter_by_mode = defaultdict(lambda: defaultdict(int))

    for tune in tune_list:
      key = tune.header['key']
      pitch, mode = key.split(' ')
      counter_by_mode[mode][pitch] += 1

    # normalize each mode
    for mode in counter_by_mode:
      total = sum(counter_by_mode[mode].values())
      for pitch in counter_by_mode[mode]:
        counter_by_mode[mode][pitch] /= total
    
    return counter_by_mode

  def get_key_diff(self, key1, key2):
    pitch_name1 = key1.split(' ')[0]
    pitch_name2 = key2.split(' ')[0]

    pitch_idx1 = self.note2idx[pitch_name1]
    pitch_idx2 = self.note2idx[pitch_name2]

    direct = pitch_idx2 - pitch_idx1
    reverse = pitch_idx2 - 12 - pitch_idx1
    higher = pitch_idx2 + 12 - pitch_idx1

    min_distance = min([abs(direct), abs(reverse), abs(higher)])
    if min_distance == abs(direct):
      return direct
    elif min_distance == abs(reverse):
      return reverse
    else:
      return higher

  def change_note_str(self, note_str, key_diff):
    pitch, dur = note_str.split('//')
    pitch = int(pitch)
    if pitch == 0: # note is rest
      return note_str
    pitch += key_diff
    return f'{pitch}//{dur}'

  def change_token(self, token, key_diff):
    main_token = token[0]
    if '//' in main_token:
      return [self.change_note_str(main_token, key_diff)] + token[1:]
    else:
      return token

  
  def get_random_key(self, org_key):
    org_key, mode = org_key.split(' ')

    new_chroma = self.chromatic.copy()
    new_chroma.remove(org_key)

    return random.choice(new_chroma) + ' ' + mode

  def get_random_stat_key(self, org_key):
    mode = org_key.split(' ')[1]
    distribution = self.key_stats[mode]

    new_key = random.choices(list(distribution.keys()), list(distribution.values()))[0]

    return new_key + ' ' + mode


  def __call__(self, str_tokens, header):
    '''
    str_tokens: list of list of str
    
    '''
    org_key = header['key']
    if self.aug_type is None:
      return str_tokens, org_key
    elif self.aug_type == 'c':
      new_key = "C" + " " + org_key.split(' ')[1]
    elif self.aug_type == 'random':
      new_key = self.get_random_key(org_key)
    elif self.aug_type == 'stat':
      new_key = self.get_random_stat_key(org_key)
    elif self.aug_type == "recover":
      if 'transcription' in  header and ' ' in header['transcription'] and header['transcription'].split(' ')[0] in self.chromatic:
        recover_key = header['transcription']
        key_diff_compen = self.chromatic.index(recover_key.split(' ')[0]) - self.chromatic.index(org_key.split(' ')[0])
        new_key = self.get_random_key(recover_key)
      else:
        return str_tokens, org_key
    else:
      print('Invalid aug_type: {}'.format(self.aug_type))
      raise NotImplementedError

    key_diff = self.get_key_diff(org_key, new_key)
    if self.aug_type == "recover":
      if 'key_diff_compen' in locals():
        key_diff += key_diff_compen
      else:
        key_diff = key_diff % 12 # always transpose to higher direction
    if key_diff > 12:
      key_diff -= 12
    if key_diff == 0:
      return str_tokens, new_key
    converted = [self.change_token(token, key_diff) for token in str_tokens]

    return converted, new_key