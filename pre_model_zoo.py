from typing import Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
# from transformers import GPT2LMHeadModel
from pre_module import MeasureGRU, DurPitchDecoder



class LanguageModel(nn.Module):
  def __init__(self, vocab_size: dict, net_param):
    super().__init__()
    self.net_param = net_param
    self.vocab_size = [x for x in vocab_size.values()]
    self.vocab_size_dict = vocab_size
    self.hidden_size = net_param.note.hidden_size
    self._make_embedding_layer()
    self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=net_param.note.num_layers, batch_first=True)
    self._make_projection_layer()

  @property
  def device(self):
      return next(self.parameters()).device

  def _make_embedding_layer(self):
    self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.net_param.emb.emb_size)

  def _make_projection_layer(self):
    self.proj = nn.Linear(self.hidden_size, self.vocab_size)
  
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      pass

  def _apply_softmax(self, logit):
    return logit.softmax(dim=-1)


  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])

    else:
      pass

    return prob

  def _prepare_start_token(self, start_token_idx):
    return torch.LongTensor([start_token_idx]).to(self.device)

  def _prepare_inference(self, start_token_idx, manual_seed):
    selected_token = self._prepare_start_token(start_token_idx)
    last_hidden = torch.zeros([self.rnn.num_layers, 1, self.rnn.hidden_size]).to(self.device)
    total_out = []
    torch.manual_seed(manual_seed)
    return selected_token, last_hidden, total_out


  def inference(self, start_token_idx=1, manual_seed=0):
    '''
    x can be just start token or length of T
    '''
    with torch.inference_mode():
      selected_token, last_hidden, total_out = self._prepare_inference(start_token_idx, manual_seed)
      while True:
        emb = self.emb(selected_token.unsqueeze(0)) # embedding vector 변환 [1,128] -> [1, 1, 128]
        hidden, last_hidden = self.rnn(emb, last_hidden)
        logit = self.proj(hidden)
        prob = torch.softmax(logit, dim=-1)
        selected_token = prob.squeeze().multinomial(num_samples=1)
        if selected_token == 2: # Generated End token
          break 
        total_out.append(selected_token)
      return torch.cat(total_out, dim=0)


class PitchDurModel(LanguageModel):
  def __init__(self, vocab_size, nn_param):
    super().__init__(vocab_size, nn_param)
    self.rnn = nn.GRU(self.net_param.emb.total_size, self.hidden_size, num_layers=nn_param.note.num_layers, batch_first=True)

  def _make_embedding_layer(self):
    self.emb = MultiEmbedding(self.vocab_size_dict, self.net_param)

  def _make_projection_layer(self):
    self.proj = nn.Linear(self.hidden_size, self.vocab_size[0] + self.vocab_size[1])

  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
      assert input_seq.ndim == 3
      emb = self.emb(input_seq)
      # main_emb = self.main_emb(input_seq.data[..., 0])
      # dur_emb = self.dur_emb(input_seq.data[..., 1])
      # emb = torch.cat([main_emb, dur_emb], dim=-1)
    return emb
  
  def _apply_softmax(self, logit):
    # logit.shape = [num_total_notes, vocab_size[0] + vocab_size[1]]
    prob = logit[:, :self.vocab_size[0]].softmax(dim=-1)
    prob = torch.cat([prob, logit[:, self.vocab_size[0]:].softmax(dim=-1)], dim=1)
    return prob

  def _sample_by_token_type(self, prob, vocab):
    main_prob = prob[:self.vocab_size[0]]
    dur_prob = prob[self.vocab_size[0]:]
    main_token = main_prob.multinomial(num_samples=1)
    if 'pitch' in vocab.vocab['main'][main_token]:
      dur_prob[:3] = 0
      dur_token = dur_prob.multinomial(num_samples=1)
    else:
      dur_token = torch.tensor([0]).to(main_prob.device)
    
    converted_out = vocab.convert_inference_token(main_token, dur_token)
    return torch.tensor(converted_out, dtype=torch.long).to(main_prob.device).unsqueeze(0)
    # return torch.cat([main_token, dur_token]).unsqueeze(0)

  def _prepare_start_token(self, start_token_idx):
    return torch.LongTensor([[start_token_idx, start_token_idx]]).to(self.device)

  def prepare_global_info(self, vocab, header):
    if header is None:
      header = vocab.get_default_header()
    header_idx = vocab.encode_header(header)

    return header, torch.LongTensor([header_idx]).to(self.device)

  def inference(self, vocab, manual_seed=0, header=None):
    header, global_condition = self.prepare_global_info(vocab, header)
    start_token_idx = vocab.vocab['main'].index('<start>')
    selected_token, last_hidden, total_out = self._prepare_inference(start_token_idx, manual_seed)
    while True:
      selected_token = torch.cat([selected_token, global_condition], dim=-1)
      emb = self._get_embedding(selected_token.unsqueeze(0)) # embedding vector 변환 [1,128] -> [1, 1, 128]
      hidden, last_hidden = self.rnn(emb, last_hidden)
      logit = self.proj(hidden)
      prob = self._apply_softmax(logit)
      selected_token = self._sample_by_token_type(prob.squeeze(), vocab)
      if 2 in selected_token: # Generated End token
        break 
      total_out.append(selected_token)
    return torch.cat(total_out, dim=0)

class MeasureInfoModel(PitchDurModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.rnn = nn.GRU(net_param.emb.total_size, 
                      net_param.note.hidden_size, 
                      num_layers=net_param.note.num_layers, 
                      dropout=net_param.note.dropout,
                      batch_first=True)

  def _make_embedding_layer(self):
    self.emb = MultiEmbedding(self.vocab_size_dict, self.net_param.emb)

  def _get_measure_info(self, measure_info, vocab):
    idx, offset = measure_info
    idx = 'm_idx:' + str(idx)
    offset = 'm_offset:' + str(offset)
    return torch.LongTensor([[vocab.tok2idx['m_idx'][idx], vocab.tok2idx['m_offset'][offset]]]).to(self.device)

  def _prepare_start_token(self, vocab, header):
    # <start>, <pad>, <m_idx:0>, <m_offset:0>
    out = vocab.prepare_start_token(header)
    return torch.LongTensor([out]).to(self.device)
    # start_token_idx = vocab.tok2idx['main']['<start>']
    # dur_token_idx =  vocab.tok2idx['dur']['<pad>']
    # m_idx = vocab.tok2idx['m_idx']['m_idx:0']
    # m_offset = vocab.tok2idx['m_offset']['m_offset:0.0']
    # return torch.LongTensor([[start_token_idx, dur_token_idx, m_idx, m_offset]]).to(self.device)

  def _inference_one_step(self, *args, **kwargs):
    selected_token, last_hidden, vocab = args
    emb = self._get_embedding(selected_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    logit = self.proj(hidden)
    prob = self._apply_softmax(logit)
    selected_token = self._sample_by_token_type(prob.squeeze(), vocab)

    return selected_token, last_hidden


  def inference(self, vocab, manual_seed=0, header=None):
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)
      start_token_idx = vocab.vocab['main'].index('<start>')
      selected_token, last_hidden, total_out = self._prepare_inference(vocab, manual_seed)
      selected_token = torch.cat([selected_token, global_condition], dim=-1)

      total_probs = []

      while True:
        selected_token, last_hidden = self._inference_one_step(selected_token, last_hidden, vocab)
        # emb = self._get_embedding(selected_token.unsqueeze(0)) # embedding vector 변환 [1,128] -> [1, 1, 128]
        # hidden, last_hidden = self.rnn(emb, last_hidden)
        # logit = self.proj(hidden)
        # prob = self._apply_softmax(logit)
        # selected_token = self._sample_by_token_type(prob.squeeze(), vocab)
        if 2 in selected_token: # Generated End token
          break 
        total_out.append(selected_token)
        # total_probs.append(prob)

        measure_sampler.update(selected_token) # update measure info
        '''
        sampled_token_str = vocab.vocab['main'][selected_token[0,0].item()]

        if '|' in sampled_token_str:
          if cur_m_offset > full_measure_duration / 2:
            # cur_m_index += 1
            # cur_m_offset = 0
            next_m_index = cur_m_index + 1
          next_m_offset = 0
          if '|1' in sampled_token_str:
            first_ending_offset = next_m_index
          if '|2' in sampled_token_str:
            next_m_index = first_ending_offset
        elif '|:' in sampled_token_str:
          next_m_index = 0
        elif '(' in sampled_token_str:
          tuplet_count = int(sampled_token_str.replace('(', ''))
        elif 'pitch' in sampled_token_str:
          sampled_dur = float(vocab.vocab['dur'][selected_token[0,1].item()].replace('dur', ''))
          # cur_m_offset += sampled_dur
          if tuplet_count == 0:
            if tuplet_duration:
              next_m_offset = cur_m_offset + tuplet_duration * 2
              tuplet_duration = 0
            else:
              next_m_offset = cur_m_offset + sampled_dur
          else:
            next_m_offset = cur_m_offset
            tuplet_count -= 1
            tuplet_duration = sampled_dur
          # print(sampled_token_str, sampled_dur, cur_m_offset)
        else:
          tuplet_count = 0
          tuplet_duration = 0
          pass
          # print(sampled_token_str, cur_m_offset)
        '''

        # measure_token = self._get_measure_info([cur_m_index, cur_m_offset], vocab)
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device)
        selected_token = torch.cat([selected_token, measure_token, global_condition], dim=-1)

        # measure_sampler.update_measure_info()
        # cur_m_offset = next_m_offset
        # cur_m_index = next_m_index

    return torch.cat(total_out, dim=0) #, torch.cat(total_probs, dim=0)


class MultiEmbedding(nn.Module):
  def __init__(self, vocab_sizes: dict, vocab_param) -> None:
    super().__init__()
    self.layers = []
    embedding_sizes = self.get_embedding_size(vocab_sizes, vocab_param)
    # if isinstance(embedding_sizes, int):
    #   embedding_sizes = [embedding_sizes] * len(vocab_sizes)
    for vocab_size, embedding_size in zip(vocab_sizes.values(), embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    # num_embeddings = torch.tensor([x.num_embeddings for x in self.layers])
    # max_indices = torch.max(x, dim=0)[0].cpu()
    # assert (num_embeddings > max_indices).all(), f'num_embeddings: {num_embeddings}, max_indices: {max_indices}'
    return torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)

  def get_embedding_size(self, vocab_sizes, vocab_param):
    embedding_sizes = [getattr(vocab_param, vocab_key) for vocab_key in vocab_sizes.keys()]
    return embedding_sizes




class MeasureHierarchyModel(MeasureInfoModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.rnn = nn.GRU(net_param.emb.total_size, 
                      net_param.note.hidden_size, 
                      num_layers=net_param.note.num_layers, 
                      dropout=net_param.note.dropout,
                      batch_first=True)
    self.measure_rnn = MeasureGRU(net_param.note.hidden_size, 
                                  net_param.measure.hidden_size,
                                  num_layers=net_param.measure.num_layers, 
                                  dropout=net_param.measure.dropout)

  def _make_projection_layer(self):
    # self.proj = DurPitchDecoder(self.hidden_size * 2, self.hidden_size, self.vocab_size[0], self.vocab_size[1])
    self.proj = nn.Linear((self.net_param.note.hidden_size + self.net_param.measure.hidden_size), self.vocab_size[0] + self.vocab_size[1])
  

  def forward(self, input_seq, measure_numbers):
    '''
    token -> rnn note_embedding                 -> projection -> pitch, duration
                  | context attention          ^
                   -> rnn measure_embedding   _|(cat)
    '''
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      
      # pitch_vec = self.time_shifted_pitch_emb(emb)
      # logit = self.proj(cat_hidden.data, pitch_vec.data) # output: [num_total_notes x vocab_size].
      logit = self.proj(cat_hidden.data)
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])

    else:
      pass

    return prob

  def _inference_one_step(self, *args, **kwargs):
    selected_token, last_hidden, last_measure_out, vocab = args
    emb = self._get_embedding(selected_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    # logit = self.proj(cat_hidden, self.emb.layers[0])
    logit = self.proj(cat_hidden)
    prob = self._apply_softmax(logit)
    selected_token = self._sample_by_token_type(prob.squeeze(), vocab)

    # assert torch.min(torch.abs(last_hidden[-1] - hidden)) < 1e-4
    return selected_token, last_hidden

  def _prepare_inference(self, vocab, header, manual_seed):
    selected_token = self._prepare_start_token(vocab, header)
    last_hidden = torch.zeros([self.rnn.num_layers, 1, self.rnn.hidden_size]).to(self.device)
    last_measure_out = torch.zeros([1, 1, self.measure_rnn.hidden_size]).to(self.device)
    last_measure_hidden = torch.zeros([self.measure_rnn.num_layers, 1, self.measure_rnn.hidden_size]).to(self.device)
    total_out = []
    torch.manual_seed(manual_seed)
    return selected_token, last_hidden, last_measure_out, last_measure_hidden, total_out

  def inference(self, vocab, manual_seed=0, header=None):
    total_hidden = []
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)

      selected_token, last_hidden, last_measure_out, last_measure_hidden, total_out = self._prepare_inference(vocab, header, manual_seed)
      selected_token = torch.cat([selected_token, global_condition], dim=-1)

      prev_measure_num = 0

      while True:
        selected_token, last_hidden = self._inference_one_step(selected_token, last_hidden, last_measure_out, vocab)
        total_hidden.append(last_hidden[-1])

        if selected_token[0,0] == 2: # Generated End token
          break 
        total_out.append(selected_token)
        # total_probs.append(prob)

        measure_sampler.update(selected_token)
        if measure_sampler.measure_number != prev_measure_num:
          last_measure_out, last_measure_hidden = self.measure_rnn.one_step(torch.cat(total_hidden, dim=0).unsqueeze(0), last_measure_hidden)
          prev_measure_num = measure_sampler.measure_number
          total_hidden = []
        
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device)
        selected_token = torch.cat([selected_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)


class MeasureNoteModel(MeasureHierarchyModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)
    self.final_rnn = nn.GRU((self.net_param.note.hidden_size + self.net_param.measure.hidden_size),
                            self.net_param.final.hidden_size,
                            num_layers=self.net_param.final.num_layers,
                            dropout=self.net_param.final.dropout,
                            batch_first=True)
    self._make_projection_layer()
  
  def _make_projection_layer(self):
    self.proj = nn.Linear(self.net_param.final.hidden_size, self.vocab_size[0] + self.vocab_size[1])

  def forward(self, input_seq, measure_numbers):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      
      final_hidden, _ = self.final_rnn(cat_hidden)

      logit = self.proj(final_hidden.data)
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])
      return prob
    else:
      raise NotImplementedError

  def _prepare_inference(self, vocab, header, manual_seed):
    selected_token, last_hidden, last_measure_out, last_measure_hidden, total_out = super()._prepare_inference(vocab, header, manual_seed)
    last_final_hidden = torch.zeros([self.final_rnn.num_layers, 1, self.final_rnn.hidden_size]).to(self.device)
    return selected_token, last_hidden, last_measure_out, last_measure_hidden, last_final_hidden, total_out
  
  def _inference_one_step(self, *args, **kwargs):
    selected_token, last_hidden, last_measure_out, last_final_hidden, vocab = args
    emb = self._get_embedding(selected_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    final_hidden, last_final_hidden = self.final_rnn(cat_hidden, last_final_hidden)
    logit = self.proj(final_hidden)
    prob = self._apply_softmax(logit)
    selected_token = self._sample_by_token_type(prob.squeeze(), vocab)
    return selected_token, last_hidden, last_final_hidden

  def inference(self, vocab, manual_seed=0, header=None):
    total_hidden = []
    with torch.inference_mode():
      header, global_condition = self.prepare_global_info(vocab, header)
      measure_sampler = MeasureSampler(vocab, header)

      selected_token, last_hidden, last_measure_out, last_measure_hidden, last_final_hidden, total_out = self._prepare_inference(vocab, header, manual_seed)
      selected_token = torch.cat([selected_token, global_condition], dim=-1)

      prev_measure_num = 0

      while True:
        selected_token, last_hidden, last_final_hidden = self._inference_one_step(selected_token, last_hidden, last_measure_out, last_final_hidden, vocab )
        total_hidden.append(last_hidden[-1])

        if selected_token[0,0] == 2: # Generated End token
          break 
        total_out.append(selected_token)
        # total_probs.append(prob)

        measure_sampler.update(selected_token)
        if measure_sampler.measure_number != prev_measure_num:
          last_measure_out, last_measure_hidden = self.measure_rnn.one_step(torch.cat(total_hidden, dim=0).unsqueeze(0), last_measure_hidden)
          prev_measure_num = measure_sampler.measure_number
          total_hidden = []
        
        measure_token = measure_sampler.get_measure_info_tensor().to(self.device)
        selected_token = torch.cat([selected_token, measure_token, global_condition], dim=-1)

    return torch.cat(total_out, dim=0)



class MeasureNotePitchFirstModel(MeasureNoteModel):
  def __init__(self, vocab_size, net_param):
    super().__init__(vocab_size, net_param)

  def _make_projection_layer(self):
    self.proj = DurPitchDecoder(self.net_param, self.vocab_size[0], self.vocab_size[1])

  def time_shifted_pitch_emb(self, emb):
    '''
    Get time-shifted pitch embedding to feed to final projection layer
    This model's projection layer first estimates pitch and then estimates duration

    '''
    end_tokens = torch.tensor([2],dtype=torch.long).to(emb.data.device) # Use 2 to represent end token
    end_vec = self.emb.layers[0](end_tokens) # pitch embedding vector 
    if isinstance(emb, PackedSequence):
      padded_emb, batch_lens = pad_packed_sequence(emb, batch_first=True)
      shifted_emb = torch.cat([padded_emb[:, 1:, :self.emb.layers[0].embedding_dim], end_vec.expand(padded_emb.shape[0], 1, -1)], dim=1)
      packed_emb = pack_padded_sequence(shifted_emb, batch_lens, batch_first=True, enforce_sorted=False)
      assert (packed_emb.sorted_indices == emb.sorted_indices).all()
      return packed_emb
    else:
      return torch.cat([emb[:, 1: :self.emb.layers[0].embedding_dim], end_vec.expand(emb.shape[0], 1, -1) ], dim=-1)

  def _inference_one_step(self, *args, **kwargs):
    selected_token, last_hidden, last_measure_out, last_final_hidden, vocab = args
    emb = self._get_embedding(selected_token.unsqueeze(0))
    hidden, last_hidden = self.rnn(emb, last_hidden)
    cat_hidden = torch.cat([hidden, last_measure_out], dim=-1)
    final_hidden, last_final_hidden = self.final_rnn(cat_hidden, last_final_hidden)
    main_token, dur_token = self.proj(final_hidden, self.emb.layers[0], vocab.pitch_range)

    converted_out = vocab.convert_inference_token(main_token, dur_token)
    selected_token = torch.tensor(converted_out, dtype=torch.long).to(emb.device).unsqueeze(0)
    return selected_token, last_hidden, last_final_hidden
    # return torch.tensor(converted_out, dtype=torch.long).to(emb.device).unsqueeze(0)
    # prob = self._apply_softmax(logit)
    selected_token = self._sample_by_token_type(prob.squeeze(), vocab)
    return selected_token, last_hidden, last_final_hidden


  def forward(self, input_seq, measure_numbers):
    '''
    token -> rnn note_embedding                 -> projection -> pitch, duration
                  | context attention          ^
                   -> rnn measure_embedding   _|(cat)
    '''
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      final_hidden, _ = self.final_rnn(cat_hidden)

      pitch_vec = self.time_shifted_pitch_emb(emb)
      logit = self.proj(final_hidden.data, pitch_vec.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])

    else:
      pass

    return prob

class MeasureSampler:
  def __init__(self, vocab, header):
    self.vocab = vocab
    self.header = header

    self.cur_m_offset = 0
    self.cur_m_index = 0

    self.tuplet_count = 0
    self.tuplet_duration = 0
    self.full_measure_duration = 8 # TODO: 박자별로 조절
    self.first_ending_offset = 0
    self.measure_number = 0
  
  def get_measure_info_tensor(self):
    idx, offset = self.cur_m_index, self.cur_m_offset
    idx = 'm_idx:' + str(idx)
    offset = 'm_offset:' + str(float(offset))

    return torch.tensor([self.vocab.encode_m_idx(idx) + self.vocab.encode_m_offset(offset, self.header)], dtype=torch.long)

    return torch.LongTensor([[self.vocab.tok2idx['m_idx'][idx], self.vocab.tok2idx['m_offset'][offset]]])

  # def update_measure_info(self):
  #   self.cur_m_offset = self.next_m_offset
  #   self.cur_m_index = self.next_m_index

  def update(self, selected_token):
    sampled_token_str = self.vocab.vocab['main'][selected_token[0,0].item()]
    if '|' in sampled_token_str:
      if self.cur_m_offset > self.full_measure_duration / 2:
        # cur_m_index += 1
        self.cur_m_index += 1
        self.measure_number += 1
      self.cur_m_offset = 0
      if '|1' in sampled_token_str:
        self.first_ending_offset = self.cur_m_index
      if '|2' in sampled_token_str:
        self.cur_m_index = self.first_ending_offset
    if '|:' in sampled_token_str:
      self.cur_m_index = 0
    elif '(3' in sampled_token_str: #TODO: Solve it with regex
      self.tuplet_count = int(sampled_token_str.replace('(', ''))
    elif 'pitch' in sampled_token_str:
      sampled_dur = float(self.vocab.vocab['dur'][selected_token[0,1].item()].replace('dur', ''))
      # cur_m_offset += sampled_dur
      if self.tuplet_count == 0:
        if self.tuplet_duration:
          self.cur_m_offset += self.tuplet_duration * 2
          self.tuplet_duration = 0
        else:
          self.cur_m_offset += sampled_dur
      else:
        self.tuplet_count -= 1
        self.tuplet_duration = sampled_dur
      # print(sampled_token_str, sampled_dur, cur_m_offset)
        if self.tuplet_count == 0:
          self.cur_m_offset += self.tuplet_duration * 2
          self.tuplet_duration = 0
    else:
      self.tuplet_count = 0
      self.tuplet_duration = 0





class MeasureGPT(MeasureInfoModel):
  def __init__(self, vocab_size, hidden_size, dropout=0.1):
    super().__init__(vocab_size, hidden_size, dropout)
    # self.rnn = 


  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      logit = self.proj(hidden.data) # output: [num_total_notes x vocab_size].
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, input_seq[1], input_seq[2], input_seq[3])

    else:
      pass

    return prob
