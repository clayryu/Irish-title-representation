from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn as nn
import torch

class MultiEmbeddingreduced(nn.Module):
  def __init__(self, vocab_sizes: dict, vocab_param, ratio) -> None:
    super().__init__()
    self.layers = []
    embedding_sizes = self.get_embedding_size(vocab_sizes, vocab_param)
    # if isinstance(embedding_sizes, int):
    #   embedding_sizes = [embedding_sizes] * len(vocab_sizes)
    for vocab_size, embedding_size in zip(vocab_sizes.values(), embedding_sizes):
      if int(embedding_size * ratio) != 0:
        self.layers.append(nn.Embedding(vocab_size, int(embedding_size * ratio)))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    # num_embeddings = torch.tensor([x.num_embeddings for x in self.layers])
    # max_indices = torch.max(x, dim=0)[0].cpu()
    # assert (num_embeddings > max_indices).all(), f'num_embeddings: {num_embeddings}, max_indices: {max_indices}'
    return torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)

  def get_embedding_size(self, vocab_sizes, vocab_param):
    embedding_sizes = [getattr(vocab_param, vocab_key) for vocab_key in vocab_sizes.keys()]
    return embedding_sizes

class ABC_meas_emb_Model(nn.Module):
  def __init__(self, trans_emb, trans_rnn, emb_size=128):
    super().__init__()
    self.hidden_size = trans_rnn.hidden_size
    self.emb = trans_emb
    self.rnn = trans_rnn
    self.proj = nn.Linear(self.hidden_size, emb_size)
    
  def forward(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      hidden, last_hidden = self.rnn(emb)
      hidden_emb = last_hidden.data[-1] # 1 x 256
      final_emb = self.proj(hidden_emb) # 1 x 128
      
    return final_emb
  
class ABC_RNN_Emb_Model(nn.Module):
  def __init__(self, trans_emb=None, trans_rnn=None, trans_measure_rnn=None, trans_final_rnn=None, net_param=None):
    super().__init__()
    self.output_size = net_param.abc_model.output_size
    self.emb = trans_emb
    self.rnn = trans_rnn
    self.measure_rnn = trans_measure_rnn
    self.final_rnn = trans_final_rnn
    self.emb_rnn = nn.GRU(input_size=512, hidden_size=net_param.emb_rnn.hidden_size, num_layers=net_param.emb_rnn.num_layers, dropout=net_param.emb_rnn.dropout, batch_first=True, bidirectional=True)
    self.hidden_size = self.emb_rnn.hidden_size * 2 * self.emb_rnn.num_layers 
    self.proj = nn.Linear(self.hidden_size, self.output_size)
    self.norm = nn.LayerNorm(self.output_size, elementwise_affine=False)
    
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      pass
    
  def forward(self, input_seq, measure_numbers):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      hidden, _ = self.rnn(emb)
      measure_hidden = self.measure_rnn(hidden, measure_numbers)

      cat_hidden = PackedSequence(torch.cat([hidden.data, measure_hidden.data], dim=-1), hidden.batch_sizes, hidden.sorted_indices, hidden.unsorted_indices)
      
      final_hidden, _ = self.final_rnn(cat_hidden)
      emb_hidden, last_emb_hidden = self.emb_rnn(final_hidden)
      # last_emb_hidden.data.shape # torch.Size([4, batch, 128])
      extr = last_emb_hidden.data.transpose(0,1) # torch.Size([batch, 4, 128])
      extr_batch = extr.reshape(len(input_seq.sorted_indices),-1) # torch.Size([batch, 512])
      batch_emb = self.norm(self.proj(extr_batch)) # torch.Size([batch, ourput_size])
      # batch_emb = batch_emb[emb_hidden.unsorted_indices] # for title matching, we need to sort the batch_emb
      return batch_emb
    else:
      raise NotImplementedError
  
class TTL_Emb_Model(nn.Module): 
    def __init__(self, in_embedding_size=1536, net_param=None):
        super().__init__()
        self.net_param = net_param
        self.hidden_size = net_param.ttl_model.hidden_size
        self.output_size = net_param.ttl_model.output_size
        self.dropout = net_param.ttl_model.dropout
        self.layer = nn.Sequential(nn.Linear(in_embedding_size, self.hidden_size*2),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.hidden_size*2, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.hidden_size, self.output_size)
                                    )
        
    def forward(self, x):
        '''
        x (torch.FloatTensor): N x Feature
        '''
        return self.layer(x)

class TTL_Emb_Model_noconfig(nn.Module): 
  def __init__(self, in_embedding_size=768):
    super().__init__()
    self.hidden_size = 256
    self.output_size = 256
    self.dropout = 0.4
    self.layer = nn.Sequential(nn.Linear(in_embedding_size, self.hidden_size*2),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.hidden_size*2, self.output_size),
                                )
      
  def forward(self, x):
    '''
    x (torch.FloatTensor): N x Feature
    '''
    return self.layer(x)
  
class Header_Emb_model(nn.Module):
  def __init__(self, vocab):
    super().__init__()
    self.hidden_size = 256
    self.output_size = 256
    self.emb_size = 64

    self.genre_emb_layer = nn.Embedding(len(vocab[0]), self.emb_size)
    self.key_emb_layer = nn.Embedding(len(vocab[1]), self.emb_size)
    self.meter_emb_layer = nn.Embedding(len(vocab[2]), self.emb_size)
    self.unit_note_length_emb_layer = nn.Embedding(len(vocab[3]), self.emb_size)

    self.layer = nn.Sequential(
                                nn.Linear(self.emb_size*4, self.hidden_size*2),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size*2, self.output_size)
                                )

  def concat_emb(self, input_tensor):
    genre_emb = self.genre_emb_layer(input_tensor[:,0].long())
    key_emb = self.key_emb_layer(input_tensor[:,1].long())
    meter_emb = self.meter_emb_layer(input_tensor[:,2].long())
    unit_note_length_emb = self.unit_note_length_emb_layer(input_tensor[:,3].long())
    return torch.cat([genre_emb, key_emb, meter_emb, unit_note_length_emb], dim=1)
  
  def forward(self, input_tensor):
    '''
    input_tensor (torch.LongTensor): N x 4
    '''
    emb = self.concat_emb(input_tensor)
    return self.layer(emb)
  
class Header_Embedding_Model_forGK(nn.Module):
  def __init__(self, vocab):
    super().__init__()
    self.hidden_size = 256
    self.output_size = 256
    self.emb_size = 128

    self.genre_emb_layer = nn.Embedding(len(vocab[0]), self.emb_size)
    self.key_emb_layer = nn.Embedding(len(vocab[1]), self.emb_size)
    # self.meter_emb_layer = nn.Embedding(len(vocab[2]), self.emb_size)
    # self.unit_note_length_emb_layer = nn.Embedding(len(vocab[3]), self.emb_size)

    self.layer = nn.Sequential(
                                nn.Linear(self.emb_size*2, self.hidden_size*2),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size*2, self.output_size)
                                )

  def concat_emb(self, input_tensor):
    genre_emb = self.genre_emb_layer(input_tensor[:,0].long())
    key_emb = self.key_emb_layer(input_tensor[:,1].long())
    # meter_emb = self.meter_emb_layer(input_tensor[:,2].long())
    # unit_note_length_emb = self.unit_note_length_emb_layer(input_tensor[:,3].long())
    return torch.cat([genre_emb, key_emb], dim=1)
    return torch.cat([genre_emb, key_emb, meter_emb, unit_note_length_emb], dim=1)
  
  def forward(self, input_tensor):
    '''
    input_tensor (torch.LongTensor): N x 4
    '''
    emb = self.concat_emb(input_tensor)
    return self.layer(emb)

class Header_Embedding_Model_forMU(nn.Module):
  def __init__(self, vocab):
    super().__init__()
    self.hidden_size = 256
    self.output_size = 256
    self.emb_size = 128

    #self.genre_emb_layer = nn.Embedding(len(vocab[0]), self.emb_size)
    #self.key_emb_layer = nn.Embedding(len(vocab[1]), self.emb_size)
    self.meter_emb_layer = nn.Embedding(len(vocab[2]), self.emb_size)
    self.unit_note_length_emb_layer = nn.Embedding(len(vocab[3]), self.emb_size)

    self.layer = nn.Sequential(
                                nn.Linear(self.emb_size*2, self.hidden_size*2),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size*2, self.output_size)
                                )

  def concat_emb(self, input_tensor):
    #genre_emb = self.genre_emb_layer(input_tensor[:,0].long())
    #key_emb = self.key_emb_layer(input_tensor[:,1].long())
    meter_emb = self.meter_emb_layer(input_tensor[:,2].long())
    unit_note_length_emb = self.unit_note_length_emb_layer(input_tensor[:,3].long())
    return torch.cat([meter_emb, unit_note_length_emb], dim=1)
    return torch.cat([genre_emb, key_emb], dim=1)
    return torch.cat([genre_emb, key_emb, meter_emb, unit_note_length_emb], dim=1)
  
  def forward(self, input_tensor):
    '''
    input_tensor (torch.LongTensor): N x 4
    '''
    emb = self.concat_emb(input_tensor)
    return self.layer(emb)

class Melody_Only_Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_size = 128
    self.output_size = 256
    self.emb_size = 128
    self.emb_layer = nn.Embedding(95, self.emb_size)
    self.rnn = nn.GRU(input_size=128, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=0.4, bidirectional=True)
    self.linear = nn.Linear(self.hidden_size*4, self.output_size)

  def forward(self, input_seq):
    '''
    input_seq (packed_seq) : seq x 20
    '''
    emb = self.emb_layer(input_seq.data[:,0]) # seq x 128
    pack_emb = nn.utils.rnn.PackedSequence(emb, input_seq.batch_sizes, input_seq.sorted_indices, input_seq.unsorted_indices)
    output, last_hidden = self.rnn(pack_emb) # output : seq x 512, last_hidden : 4 x batch x 128
    batch_last_hidden = last_hidden.permute(1,0,2).reshape(-1, self.hidden_size*4) # batch x 512
    return self.linear(batch_last_hidden)

# cnn layer 파라미터의 개수를 구해보자. 파라미터의 개수를 알아야 지금 모델이 내가 감당할 만한 모델인지 알 수 있다.
# kernel의 목적은 보고 있는 timestep 혹은 픽셀 정보를 1개의 뉴런으로 바꾸어주는 곱연산이다. 5 to 1
# 그리고 채널이란 독립적으로 존재하는 timestep 혹은 픽셀의 정보다. 이는 종합적으로 사용이 될 개별 정보로서 채널은 절대로 연속된 정보가 아니다.
# 따라서 kernel size x channel size
class ABC_CNN_Emb_Model(nn.Module):
  def __init__(self, trans_emb=None, vocab_size=None, net_param=None, pre_param=None, emb_ratio=1):
    super().__init__()
    self.output_size = net_param.abc_model.output_size
    self.hidden_size = net_param.abc_model.hidden_size
    
    self.dropout = net_param.abc_conv.dropout
    self.num_layers = len(net_param.abc_conv.in_channels)
    self.in_channels = net_param.abc_conv.in_channels
    self.out_channels = net_param.abc_conv.out_channels
    self.kernel_size = net_param.abc_conv.kernel_sizes
    self.stride = net_param.abc_conv.strides
    self.padding = net_param.abc_conv.paddings
    self._make_conv_layer()

    self.emb_ratio = emb_ratio
    if vocab_size is not None and net_param is not None and trans_emb is None:
      self.vocab_size_dict = vocab_size
      self.net_param = net_param
      self.pre_param = pre_param
      self._make_embedding_layer()
    elif trans_emb is not None:
      self.emb = trans_emb
    
    self.emb_total_list = [x.embedding_dim for x in self.emb.layers]
    self.emb_total_size = sum(self.emb_total_list)

    self.linear_layer1 = nn.Sequential(
      nn.Linear(self.emb_total_size, self.hidden_size)
    )

    self.linear_layer2 = nn.Sequential(
      nn.Linear(self.hidden_size, self.output_size)
    )

  def _make_conv_layer(self):
    for idx in range(self.num_layers):
      setattr(self, f'conv{idx}', nn.Conv1d(self.in_channels[idx], self.out_channels[idx], self.kernel_size[idx], self.stride[idx], self.padding[idx]))
      print(f'conv{idx} : {self.in_channels[idx]} {self.out_channels[idx]} {self.kernel_size[idx]} {self.stride[idx]} {self.padding[idx]}')
      setattr(self, f'batch_norm{idx}', nn.BatchNorm1d(self.out_channels[idx]))
      setattr(self, f'relu{idx}', nn.ReLU())
      setattr(self, f'dropout{idx}', nn.Dropout(self.dropout))
    
  def _make_embedding_layer(self):
    self.emb = MultiEmbeddingreduced(self.vocab_size_dict, self.pre_param.emb, self.emb_ratio)
    
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
      return emb
    else:
      pass
    
  def forward(self, input_seq, measure_numbers):
    if isinstance(input_seq, PackedSequence):
      emb = self._get_embedding(input_seq)
      unpacked_emb, _ = pad_packed_sequence(emb, batch_first=True)
      #unpacked_emb = unpacked_emb.transpose(1,2)

      for_conv = self.linear_layer1(unpacked_emb)
      for_conv = for_conv.transpose(1,2)
      for idx in range(self.num_layers):
        conv = getattr(self, f'conv{idx}')
        batch_norm = getattr(self, f'batch_norm{idx}')
        relu = getattr(self, f'relu{idx}')
        dropout = getattr(self, f'dropout{idx}')
        if idx == len(self.num_layers) - 1:
          for_conv = relu(batch_norm(conv(for_conv)))
        else:
          for_conv = dropout(relu(batch_norm(conv(for_conv))))
      after_conv = nn.AdaptiveMaxPool1d(1)(for_conv)

      before_linear = after_conv.view(after_conv.size(0), -1)
      batch_emb = self.linear_layer2(before_linear)
      
      return batch_emb
    else:
      raise NotImplementedError
    
class PromptEncoder(torch.nn.Module):
  def __init__(self, template, hidden_size, tokenizer, device, args):
    super().__init__()
    self.device = device
    self.spell_length = sum(template)
    self.hidden_size = hidden_size
    self.tokenizer = tokenizer
    self.args = args
    # ent embedding
    self.cloze_length = template
    self.cloze_mask = [
      [1] * self.cloze_length[0] # first cloze
      + [1] * self.cloze_length[1] # second cloze
      + [1] * self.cloze_length[2] # third cloze
    ]
    self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
  
    self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
    # embeding
    self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
    # LSTM
    self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size//2,
                                   num_layers=2,
                                   dropout=0.5,
                                   bidirectional=True,
                                   batch_first=True)
    self.mlp_head = torch.nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                        self.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size))
    