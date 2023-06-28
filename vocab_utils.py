import json
from collections import defaultdict

import torch
from pre_str_utils import split_note

from typing import List, Dict, Tuple, Union, Optional, Any

class TokenVocab:
  def __init__(self, json_path, unique_char_set=None) -> None:
    self.key_types = ['main', 'dur',  'm_idx', 'm_offset', 'key', 'meter', 'unit_length', 'rhythm']

    if unique_char_set is None:
      with open(json_path, 'r') as f:
        self.vocab = json.load(f)
    else:
      self.vocab = defaultdict(list)
      self.vocab['rhythm'] = ['R: Unspecified']
      for token in unique_char_set:
        if 'dur' in token:
          self.vocab['dur'].append(token)
        elif 'K:' in token:
          self.vocab['key'].append(token)
        elif 'M:' in token:
          self.vocab['meter'].append(token)
        elif 'L:' in token:
          self.vocab['unit_length'].append(token)
        elif 'R:' in token:
          self.vocab['rhythm'].append(token)
        elif 'm_idx:' in token:
          self.vocab['m_idx'].append(token)
        elif 'm_offset:' in token:
          self.vocab['m_offset'].append(token)
        else:
          if token != '<start>':
            self.vocab['main'].append(token)
      
      for key in self.vocab:
        if key in ['main', 'dur', 'm_idx', 'm_offset']:
          self.vocab[key] = ['<pad>', '<start>', '<end>'] + sorted(self.vocab[key])
      
      self.vocab['main'] = self.augment_pitch_vocab()
      self.vocab['key'] = self.augment_key_vocab()
    self.tok2idx = {key: {k:i for i, k in enumerate(value)}  for key, value in self.vocab.items() }
      
    if unique_char_set is not None and json_path is not None:
      self.save_json(json_path)


  def augment_pitch_vocab(self):
    main_vocab = self.vocab['main']
    pitch_vocab = [x for x in main_vocab if 'pitch' in x]
    pitch_int = [int(x.replace('pitch', '')) for x in pitch_vocab if x.replace('pitch', '').isdigit()]
    if 0 in pitch_int:
      pitch_int.remove(0)
    min_pitch = min(pitch_int)
    max_pitch = max(pitch_int)
    new_pitch_vocab = [f'pitch{x}' for x in range(min_pitch-7, max_pitch+24)]

    new_main_vocab = [x for x in main_vocab if x not in new_pitch_vocab] + new_pitch_vocab
    new_main_vocab =  ['<pad>', '<start>', '<end>'] + sorted([x for x in new_main_vocab if x not in ['<start>', '<pad>', '<end>'] ])
    return new_main_vocab

  def augment_key_vocab(self):
    key_vocab = self.vocab['key']
    new_key_vocab = []
    modes = set([x.split(':')[1].lstrip().split(' ')[1] for x in key_vocab if 'K:' in x])
    chromatic = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    for mode in modes:
      for chrom in chromatic:
        new_key_vocab.append(f'K: {chrom} {mode}')

    new_key_vocab = [x for x in key_vocab if x not in new_key_vocab] + new_key_vocab
    return sorted(new_key_vocab)

  def get_size(self):
    # return [len(self.vocab[x]) for x in self.key_types]
    return {x:len(self.vocab[x]) for x in self.key_types}

  def __len__(self):
    return len(self.vocab['main']) + len(self.vocab['dur'])

  def save_json(self, json_path):
    with open(json_path, 'w') as f:
      json.dump(dict(self.vocab), f, indent=2, ensure_ascii=False)

  def __call__(self, word, header=None):
    if isinstance(word, list) and len(word) == 3: # token, measure_idx, measure_offset
      wrd, m_idx, m_offset = word
      return self(wrd) + self.encode_m_idx(m_idx) + self.encode_m_offset(m_offset, header)
    else:
      if word in self.tok2idx['main']:
        return [self.tok2idx['main'][word], 0]
      elif '//' in word:
        pitch, dur = split_note(word)
        return [self.tok2idx['main'][pitch], self.tok2idx['dur'][dur]]
      else:
        print('Error: {} is not in the vocab'.format(word))

  def encode_m_idx(self, m_idx):
    return [self.tok2idx['m_idx'][m_idx]]
  
  def encode_m_offset(self, m_offset, header):
    return [self.tok2idx['m_offset'][m_offset]]

  def decode(self, alist):
    if isinstance(alist, torch.Tensor):
      alist = alist.tolist()
    if isinstance(alist[0], list):
      return [self.decode(x) for x in alist]
    else:
      main_word = self.vocab['main'][alist[0]]
      if 'pitch' in main_word: # pad, start, end. In this case, alist[0] is the token in the main_vocab
        return self.vocab['main'][alist[0]] + '//' + self.vocab['dur'][alist[1]]
      else:
        return self.vocab['main'][alist[0]]

  def encode_header(self, header):
    if 'rhythm' not in header:
      header['rhythm'] = 'Unspecified'
    return [self.tok2idx['key']['K: ' + header['key']], 
            self.tok2idx['meter']['M: ' + header['meter']], 
            self.tok2idx['unit_length']['L: ' + header['unit note length']], 
            self.tok2idx['rhythm']['R: ' + header['rhythm']]]

  def get_default_header(self):
    default_header = {'key':'C Major', 'meter':'4/4', 'unit note length':'1/8', 'rhythm':'reel'}
    # if 'K: '+ default_header['key'] not in self.vocab['key']:
    #   if 'K: G' in self.tok2idx['key']:
    #     default_header['key'] = 'G'
    #   elif 'K: Gmaj' in self.tok2idx['key']:
    #     default_header['key'] = 'Gmaj'
    
    if 'R: ' + default_header['rhythm'] not in self.vocab['rhythm']:
      if 'R: Reel' in self.tok2idx['rhythm']:
        default_header['rhythm'] = 'Reel'
      elif 'R: reel' in self.tok2idx['rhythm']:
        default_header['rhythm'] = 'reel'
      else:
        default_header['rhythm'] = 'Unspecified'

    return default_header

  def prepare_start_token(self, header):
    start_token_idx = self.tok2idx['main']['<start>']
    dur_token_idx =  self.tok2idx['dur']['<pad>']
    m_idx = self.tok2idx['m_idx']['m_idx:0']
    m_offset = self.tok2idx['m_offset']['m_offset:0.0']
    return [start_token_idx, dur_token_idx, m_idx, m_offset]


  def convert_inference_token(self, main_token, dur_token):
    return [main_token.item(), dur_token.item()]

  @property
  def pitch_range(self):
    return [i for i, value in enumerate(self.vocab['main']) if 'pitch' in value]


class MusicTokenVocab(TokenVocab):
  def __init__(self, json_path, unique_char_set=None) -> None:
    super().__init__(json_path, unique_char_set)


    self.vocab['root'], self.vocab['mode'] = self.make_root_and_modes()
    self.mode2key = {'Major': 0, 'minor': -3, 'Ionian': 0, 'Aeolian': -3,
               'Mixolydian': -1, 'Dorian': -2, 'Phrygian': -4, 'Lydian': 1, 'Locrian': -5}
    self.root2key = {'C#': 7, 'F#': 6, 'B': 5, 'E': 4, 'A': 3, 'D': 2, 'G': 1, 'C': 0,
           'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}
    self.vocab['key_sig'] = [str(x) for x in range(-9, 8)]

    self.vocab['numer'], self.vocab['denom'] = self.make_numer_and_denoms()
    self.vocab['is_compound'] = [False, True]
    self.vocab['is_triple'] = [False, True]

    self.key_types += ['root', 'mode', 'key_sig',  'numer', 'denom','is_compound', 'is_triple']


    self.tok2idx = {key: {k:i for i, k in enumerate(value)}  for key, value in self.vocab.items() }

    if unique_char_set is not None and json_path is not None:
      self.save_json(json_path)


  def parse_key(self, key):
    # example: key = G Major
    root, mode = key.split(' ')

    # get key signature
    root_calib = self.root2key[root]
    mode_calib = self.mode2key[mode]

    key_sig = str(root_calib + mode_calib)

    return root, mode, key_sig
  
  def parse_meter(self, meter):
    # example: meter = 4/4
    numer, denom = meter.split('/')
    is_compound = int(numer) in [6, 9, 12]
    is_triple = int(numer) in [3, 9]
    return numer, denom, is_compound, is_triple

  def make_root_and_modes(self):
    key_list = [x for x in self.vocab['key'] if 'K:' in x]
    root_note = [x.split(':')[1].lstrip().split(' ')[0] for x in key_list]
    root_note = sorted(list(set(root_note)))

    mode_list = [x.split(':')[1].lstrip().split(' ')[1] for x in key_list]
    mode_list = sorted(list(set(mode_list)))

    return root_note, mode_list

  def make_numer_and_denoms(self):
    meter_list = [x for x in self.vocab['meter'] if 'M:' in x]
    denom_list = [x.split(':')[1].lstrip().split('/')[1] for x in meter_list]
    denom_list = sorted(list(set(denom_list)))
    numer_list = [x.split(':')[1].lstrip().split('/')[0] for x in meter_list]
    numer_list = sorted(list(set(numer_list)))

    return numer_list, denom_list


  def encode_header(self, header):
    if 'rhythm' not in header:
      header['rhythm'] = 'Unspecified'

    main_list =  [self.tok2idx['key']['K: ' + header['key']], 
            self.tok2idx['meter']['M: ' + header['meter']], 
            self.tok2idx['unit_length']['L: ' + header['unit note length']], 
            self.tok2idx['rhythm']['R: ' + header['rhythm']]]

    root, mode, key_sig = self.parse_key(header['key'])
    key_list = [self.tok2idx['root'][root], self.tok2idx['mode'][mode], self.tok2idx['key_sig'][key_sig]]

    numer, denom, is_compound, is_triple = self.parse_meter(header['meter'])
    meter_list = [self.tok2idx['numer'][numer], self.tok2idx['denom'][denom], self.tok2idx['is_compound'][is_compound], self.tok2idx['is_triple'][is_triple]]

    main_list += key_list + meter_list

    return main_list

class NoteMusicTokenVocab(MusicTokenVocab):
  def __init__(self, json_path, unique_char_set=None):
    super().__init__(json_path, unique_char_set)

    self.vocab['pitch_class'] = ['<pad>', '<start>', '<end>'] + [x for x in range(12)]
    self.vocab['octave'] = ['<pad>', '<start>', '<end>'] + [x for x in range(12)]
    self.vocab['m_idx_mod4'] = ['<pad>', '<start>', '<end>'] + [x for x in range(4)]
    self.vocab['is_onbeat'] = ['<pad>', '<start>', '<end>'] + [False, True]
    self.vocab['is_middle_beat'] = ['<pad>', '<start>', '<end>'] + [False, True]

    self.key_types = ['main', 'dur',
                      'pitch_class', 'octave', 
                      'm_idx', 'm_idx_mod4', 
                      'm_offset', 'is_onbeat', 'is_middle_beat',
                      'key', 'meter', 'unit_length', 'rhythm']
    self.key_types += ['root', 'mode', 'key_sig',  'numer', 'denom','is_compound', 'is_triple']
    self.tok2idx = {key: {k:i for i, k in enumerate(value)}  for key, value in self.vocab.items() }

    if unique_char_set is not None and json_path is not None:
      self.save_json(json_path)

  def encode_m_idx(self, m_idx) -> List[int]:
    if 'm_idx:' in m_idx:
      value = m_idx.split(':')[1]
      if value == "None":
        return [self.tok2idx['m_idx'][m_idx], 0]
      value = int(m_idx.split(':')[1])   
      mod_4 = value % 4
      return [self.tok2idx['m_idx'][m_idx], self.tok2idx['m_idx_mod4'][mod_4]]
    else: # <pad>, <start>, <end>
      return [self.tok2idx['m_idx'][m_idx], self.tok2idx['m_idx_mod4'][m_idx]]

  def encode_m_offset(self, m_offset, header) -> List[int]:
    if 'm_offset:' in m_offset:
      meter = header['meter']
      unit = header['unit note length']
      unit = int(unit.split('/')[1].strip())
      numer, denom, is_compound, is_triple = self.parse_meter(meter)
      numer = int(numer)
      denom = int(denom)

      unit_beat = unit / denom
      
      value = m_offset.split(':')[1]
      if value == "None":
        idx = self.tok2idx['m_offset'][m_offset]
        return [idx, 0, 0]
      value = float(value)
      middle_beat = []
      if numer == 4:
        on_beat_offset = [0, 1, 2, 3]
        middle_beat = [2]
      elif numer == 2:
        on_beat_offset = [0, 1]
      elif numer == 3:
        on_beat_offset = [0, 1, 2]
      elif numer == 6:
        on_beat_offset = [0, 3]
      elif numer == 9:
        on_beat_offset = [0, 3, 6]
      elif numer == 12:
        on_beat_offset = [0, 3, 6, 9]
        middle_beat = [6]
      else:
        on_beat_offset = [0]

      on_beat_offset = [x * unit_beat for x in on_beat_offset]
      middle_beat = [x * unit_beat for x in middle_beat]

      is_onbeat = value in on_beat_offset
      is_middle_beat = value in middle_beat

      return [self.tok2idx['m_offset'][m_offset], 
              self.tok2idx['is_onbeat'][is_onbeat], 
              self.tok2idx['is_middle_beat'][is_middle_beat]]

    else:
      value = self.tok2idx['m_offset'][m_offset]
      return [value, 0, 0]

  def encode_pitch(self, pitch:str) -> List[int]:
    assert 'pitch' in pitch
    pitch_idx = self.tok2idx['main'][pitch]
    value = int(pitch.replace('pitch',''))

    if value == 0:
      return [pitch_idx, 0, 0]
    else:
      pitch_class = value % 12
      octave = value // 12
      return [pitch_idx, self.tok2idx['pitch_class'][pitch_class], self.tok2idx['octave'][octave]]

  def prepare_start_token(self, header):
    main = '<start>'
    m_idx = 'm_idx:0'
    m_offset = 'm_offset:0.0'

    return self([main, m_idx, m_offset], header)

  def convert_inference_token(self, main_token, dur_token):
    main_idx = main_token.item()
    dur_idx = dur_token.item()

    word = self.vocab['main'][main_idx]
    if 'pitch' in word:
      pitch_idx, pitch_class_idx, octave_idx = self.encode_pitch(word)
      return [pitch_idx, dur_idx, pitch_class_idx, octave_idx]
    else:
      return [main_idx, dur_idx, 0, 0]

  def __call__(self, word, header=None):
    if isinstance(word, list) and len(word) == 3: # token, measure_idx, measure_offset
      wrd, m_idx, m_offset = word
      return self(wrd) + self.encode_m_idx(m_idx) + self.encode_m_offset(m_offset, header)
    else:
      if word in self.tok2idx['main']:
        return [self.tok2idx['main'][word], 0, 0, 0]
      elif '//' in word:
        pitch, dur = split_note(word)
        main, pitch_class, octave = self.encode_pitch(pitch)
        dur = self.tok2idx['dur'][dur]
        return [main, dur, pitch_class, octave]
        # return self.encode_pitch(pitch) + [self.tok2idx['dur'][dur]]
      else:
        print('Error: {} is not in the vocab'.format(word))
        return None

class Vocab_dictionary:
  def __init__(self, vocab):
    self.vocab = vocab
    self.vocab_dict = {}
    self.encode_m_idx_dict = {}

  def encode_m_idx(self, m_idx):
    if m_idx not in self.encode_m_idx_dict:
      self.encode_m_idx_dict[m_idx] = self.vocab.encode_m_idx(m_idx)
    return self.encode_m_idx_dict[m_idx]

  def __call__(self, token):
    if token not in self.vocab_dict:
      self.vocab_dict[token] = self.vocab(token)
    return self.vocab_dict[token]