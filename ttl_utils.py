from collections import defaultdict
from pathlib import Path
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.nn import ConstantPad1d
import json
import time
import random
import math

from pyabc import pyabc

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
  return text

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
  return True

def prepare_abc_title(paths: list):
    delete_list = ['Z:', 'F:', 'W:'] # F: 가 들어간 abc notation이 있는지 확인 -> 일단은 없음
    tune_list = []
    error_list = []
    filtered_tunes = []
    filtered_tunes_list = []
    title_in_text = []
    
    for path in paths:
        f = open(path)
        abc = f.readlines()
        length = len(abc)

        for line in reversed(abc):
          length -= 1
          if line[:2] in delete_list: # 지워야할 헤더 항목과 각 라인의 앞 부분이 일치하면 pop
            abc.pop(length)

        abc = ''.join(abc)
        abc = abc.replace('\\\n', '\n') # escape 문자로 \ 하나를 더 붙인 부분을 그냥 줄바꿈 기호로 치환

        try: # TODO: 같은 tunes에 묶인 tune을 필요시 구별해서 묶어야함
          tunes = pyabc.Tunes(abc=abc)
          filtered_tunes = []
          for tune in tunes.tunes:
            # tune = pyabc.Tune(abc=abc)
            if 'rhythm' not in tune.header:
              tune.header['rhythm'] = 'Unspecified'
            if 'unit note length' not in tune.header:
              tune.header['rhythm'] = '1/8'
            if is_valid_tune(tune):
              tune_list.append(tune)
              filtered_tunes.append(tune)
          
        except:
          error_list.append(path.name)
          
        if len(filtered_tunes) > 0:
          filtered_tunes_list.append(filtered_tunes)
          title_in_text.append(filtered_tunes[0].title)
          
    return tune_list, error_list, filtered_tunes_list, title_in_text

def pack_collate_title(raw_batch:list):
    
    melody = [mel_pair[0] for mel_pair in raw_batch]
    title = [pair[1] for pair in raw_batch] #pair[1] 
    
    packed_melody = pack_sequence(melody, enforce_sorted=False)
    #packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
    
    if len(raw_batch[0]) == 2:
      return packed_melody, torch.stack(title, dim=0)
    elif len(raw_batch[0]) == 3:
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
      packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
      return packed_melody, torch.stack(title, dim=0), packed_measure_numbers
    else:
      raise ValueError("Unknown raw_batch format")
    
def pack_collate_title_sampling_train(raw_batch:list, sample_num=30):
    
  melody = []
  title = []
  measure_numbers = []
  if sample_num is not None:
    for mel_pair in raw_batch:
      if len(mel_pair[0]) <= sample_num: # shorter than the sample_num then pad
        pad_size = sample_num - mel_pair[0].size()[0]
        RU_num = math.ceil(pad_size/2)
        RD_num = math.floor(pad_size/2)

        padded_mel = ConstantPad1d((RD_num, RU_num), 0)(mel_pair[0].T)
        melody.append(padded_mel.T)
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
      else: # longer than the sample_num then sample
        sampled_num = random.randint(0,len(mel_pair[0])-sample_num)
        #sampled_num = 0
        melody.append(mel_pair[0][sampled_num:sampled_num+sample_num])
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
  else:
    melody = [mel_pair[0] for mel_pair in raw_batch]
    title = [pair[1] for pair in raw_batch] #pair[1] 
    if len(raw_batch[0]) == 3:
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
  
  packed_melody = pack_sequence(melody, enforce_sorted=False)
  #packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
  if len(raw_batch[0]) == 2:
    return packed_melody, torch.stack(title, dim=0)
  elif len(raw_batch[0]) == 4: # melody, ttlemb, measure_numbers, textttl
    packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
    return packed_melody, torch.stack(title, dim=0), packed_measure_numbers
  else:
    raise ValueError("Unknown raw_batch format")
  
def pack_collate_title_sampling_valid(raw_batch:list, sample_num=30):
  #random.seed(42)
  torch.manual_seed(42)
  melody = []
  title = []
  measure_numbers = []
  
  if sample_num is not None:
    for mel_pair in raw_batch:
      if len(mel_pair[0]) <= sample_num:
        pad_size = sample_num - mel_pair[0].size()[0]
        RU_num = math.ceil(pad_size/2)
        RD_num = math.floor(pad_size/2)

        padded_mel = ConstantPad1d((RD_num, RU_num), 0)(mel_pair[0].T)
        melody.append(padded_mel.T)
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
      else:
        # sampled_num = random.randint(0,len(mel_pair[0])-sample_num)
        sampled_num = torch.randint(0,len(mel_pair[0])-sample_num, (1,)).item()
        #sampled_num = 0
        melody.append(mel_pair[0][sampled_num:sampled_num+sample_num])
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
  else:
    melody = [mel_pair[0] for mel_pair in raw_batch]
    title = [pair[1] for pair in raw_batch] #pair[1] 
    if len(raw_batch[0]) == 3:
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
  
  packed_melody = pack_sequence(melody, enforce_sorted=False)
  #packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
  
  if len(raw_batch[0]) == 2:
    return packed_melody, torch.stack(title, dim=0)
  elif len(raw_batch[0]) == 4: # melody, ttlemb, measure_numbers, textttl
    packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
    return packed_melody, torch.stack(title, dim=0), packed_measure_numbers
  else:
    raise ValueError("Unknown raw_batch format")

def pack_collate_title_sampling_textttl(raw_batch:list, sample_num=None):
  #random.seed(42)
  torch.manual_seed(42)
  melody = []
  title = []
  measure_numbers = []
  textttl = []
  
  if sample_num is not None:
    for mel_pair in raw_batch:
      if len(mel_pair[0]) <= sample_num:
        pad_size = sample_num - mel_pair[0].size()[0]
        RU_num = math.ceil(pad_size/2)
        RD_num = math.floor(pad_size/2)

        padded_mel = ConstantPad1d((RD_num, RU_num), 0)(mel_pair[0].T)
        melody.append(padded_mel.T)
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
        textttl.append(mel_pair[3])
      else:
        # sampled_num = random.randint(0,len(mel_pair[0])-sample_num)
        sampled_num = torch.randint(0,len(mel_pair[0])-sample_num, (1,)).item()
        #sampled_num = 0
        melody.append(mel_pair[0][sampled_num:sampled_num+sample_num])
        title.append(mel_pair[1])
        measure_numbers.append(mel_pair[2])
        textttl.append(mel_pair[3])
  else:
    melody = [mel_pair[0] for mel_pair in raw_batch]
    title = [pair[1] for pair in raw_batch] #pair[1] 
    measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
    textttl = [mel_pair[3] for mel_pair in raw_batch]
  
  packed_melody = pack_sequence(melody, enforce_sorted=False)
  #packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
  
  if len(raw_batch[0]) == 2:
    return packed_melody, torch.stack(title, dim=0)
  elif len(raw_batch[0]) == 4:
    packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
    return packed_melody, torch.stack(title, dim=0), packed_measure_numbers, textttl
  else:
    raise ValueError("Unknown raw_batch format")
  
class Pack_collate_sampling():
  def __init__(self, sample_num=30):
    self.sample_num = sample_num
    self.melody = []
    self.title = []
    self.measure_numbers = []
    self.textttl = []

  def __call__(self, raw_batch:list):
    if self.sample_num != 'None':
      for mel_pair in raw_batch:
        if len(mel_pair[0]) <= self.sample_num:
          pad_size = self.sample_num - mel_pair[0].size()[0]
          RU_num = math.ceil(pad_size/2)
          RD_num = math.floor(pad_size/2)

          padded_mel = ConstantPad1d((RD_num, RU_num), 0)(mel_pair[0].T)
          melody.append(padded_mel.T)
          title.append(mel_pair[1])
          measure_numbers.append(mel_pair[2])
          textttl.append(mel_pair[3])
        else:
          # sampled_num = random.randint(0,len(mel_pair[0])-sample_num)
          sampled_num = torch.randint(0,len(mel_pair[0])-self.sample_num, (1,)).item()
          #sampled_num = 0
          melody.append(mel_pair[0][sampled_num:sampled_num+self.sample_num])
          title.append(mel_pair[1])
          measure_numbers.append(mel_pair[2])
          textttl.append(mel_pair[3])
    else:
      melody = [mel_pair[0] for mel_pair in raw_batch]
      title = [pair[1] for pair in raw_batch] #pair[1] 
      measure_numbers = [mel_pair[2] for mel_pair in raw_batch]
      textttl = [mel_pair[3] for mel_pair in raw_batch]
    
    packed_melody = pack_sequence(melody, enforce_sorted=False)
    #packed_shifted_melody = pack_sequence(shifted_melody, enforce_sorted=False)
    
    if len(raw_batch[0]) == 2:
      return packed_melody, torch.stack(title, dim=0)
    elif len(raw_batch[0]) == 4:
      packed_measure_numbers = pack_sequence(measure_numbers, enforce_sorted=False)
      return packed_melody, torch.stack(title, dim=0), packed_measure_numbers, textttl
    else:
      raise ValueError("Unknown raw_batch format")
    
def metric_calculate(header_tensor, ttl_tensor):
  cos_sim = torch.matmul(ttl_tensor, header_tensor.T)
  header_norm = torch.norm(header_tensor, dim=1)
  ttl_norm = torch.norm(ttl_tensor, dim=1)
  cos_sim = cos_sim / header_norm / ttl_norm.unsqueeze(1)
  cos_sim = cos_sim.cpu().detach().numpy()
  rank_matrix = np.argsort(cos_sim, axis=1)
  rank_matrix = np.flip(rank_matrix, axis=1)
  mrr_dict = {idx-1:1/idx for idx in range(1, len(rank_matrix)+1)}
  dcg_dict = {idx-1:1/np.log2(idx+1) for idx in range(1, len(rank_matrix)+1)}
  sum_mrr = 0
  sum_dcg = 0
  for idx in range(len(rank_matrix)):
    position = np.argwhere(rank_matrix[idx]==idx).item()
    mrr_score = mrr_dict[position]
    dcg_score = dcg_dict[position]
    sum_mrr += mrr_score
    sum_dcg += dcg_score
  return sum_mrr/len(rank_matrix), sum_dcg/len(rank_matrix)