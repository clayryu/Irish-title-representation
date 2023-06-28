import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



def find_boundaries(diff_boundary, higher_indices, i):
  '''
  diff_boundary (torch.Tensor): N x T
  beat_numbers (torch.Tensor): zero_padded N x T
  i (int): batch index
  '''
  out = [0] + (diff_boundary[diff_boundary[:,0]==i][:,1]+1 ).tolist() + [torch.max(torch.nonzero(higher_indices[i])).item()+1]
  if out[1] == 0: # if the first boundary occurs in 0, it will be duplicated
    out.pop(0)
  return out

def find_boundaries_batch(beat_numbers):
  '''
  beat_numbers (torch.Tensor): zero_padded N x T
  '''
  diff_boundary = torch.nonzero(beat_numbers[:,1:] - beat_numbers[:,:-1] == 1).cpu()
  return [find_boundaries(diff_boundary, beat_numbers, i) for i in range(len(beat_numbers))]


def get_softmax_by_boundary(similarity, boundaries, fn=torch.softmax):
  '''
  similarity = similarity of a single sequence of data (T x C)
  boundaries = list of a boundary index (T)
  '''
  return  [fn(similarity[boundaries[i-1]:boundaries[i],: ], dim=0)  \
              for i in range(1, len(boundaries))
                if boundaries[i-1] < boundaries[i] # sometimes, boundaries can start like [0, 0, ...]
          ]


def run_hierarchy_rnn_with_pack(sequence, rnn):
  '''
  sequence (torch.Tensor): zero-padded sequece of N x T x C
  lstm (torch.LSTM): LSTM layer
  '''
  batch_note_length = sequence.shape[1] - (sequence==0).all(dim=-1).sum(-1)
  packed_sequence = pack_padded_sequence(sequence, batch_note_length.cpu(), True, False )
  hidden_out, _ = rnn(packed_sequence)
  hidden_out, _ = pad_packed_sequence(hidden_out, True)

  return hidden_out


def make_higher_node(lower_out, attention_weights, lower_indices, higher_indices, lower_is_note=False):
    # higher_nodes = []

    similarity = attention_weights.get_attention(lower_out)
    if lower_is_note:
        boundaries = find_boundaries_batch(higher_indices)
    else:
        higher_boundaries = find_boundaries_batch(higher_indices)
        zero_shifted_lower_indices = lower_indices - lower_indices[:,0:1]
        num_zero_padded_element_by_batch = ((lower_out!=0).sum(-1)==0).sum(1)
        len_lower_out = (lower_out.shape[1] - num_zero_padded_element_by_batch).tolist()
        boundaries = [zero_shifted_lower_indices[i, higher_boundaries[i][:-1]].tolist() + [len_lower_out[i]] for i in range(len(lower_out))]

        # higher_boundaries = [0] + (torch.where(higher_indices[1:] - higher_indices[:-1] == 1)[0] + 1).cpu().tolist() + [len(higher_indices)]
        # boundaries = [int(lower_indices[x]-lower_indices[0]) for x in higher_boundaries[:-1]] + [lower_out.shape[-2]]
    
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(
      [torch.cat(get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx]))
        for batch_idx in range(len(lower_out))], 
      batch_first=True
    )
    # softmax_similarity = torch.cat([torch.softmax(similarity[:,boundaries[i-1]:boundaries[i],:], dim=1) for i in range(1, len(boundaries))], dim=1)
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(lower_out.split(split_size=attention_weights.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], lower_out.shape[-1])
        higher_nodes = torch.nn.utils.rnn.pad_sequence([
          torch.cat([torch.sum(weighted_x[i:i+1,boundaries[i][j-1]:boundaries[i][j],: ], dim=1) for j in range(1, len(boundaries[i]))], dim=0) \
          for i in range(len(lower_out))
        ], batch_first=True
        )
    else:
        weighted_sum = softmax_similarity * lower_out
        higher_nodes = torch.cat([torch.sum(weighted_sum[:,boundaries[i-1]:boundaries[i],:], dim=1) 
                                for i in range(1, len(boundaries))]).unsqueeze(0)
    return higher_nodes


def span_beat_to_note_num(beat_out, beat_number):
  '''
  beat_out (torch.Tensor): N x T_beat x C
  beat_number (torch.Tensor): N x T_note x C
  '''
  zero_shifted_beat_number = beat_number - beat_number[:,0:1]
  len_note = cal_length_from_padded_beat_numbers(beat_number)

  batch_indices = torch.cat([torch.ones(length)*i for i, length in enumerate(len_note)]).long()
  note_indices = torch.cat([torch.arange(length) for length in len_note])
  beat_indices = torch.cat([zero_shifted_beat_number[i,:length] for i, length in enumerate(len_note)]).long()

  beat_indices = beat_indices - 1 # note has to get only previous beat info

  span_mat = torch.zeros(beat_number.shape[0], beat_number.shape[1], beat_out.shape[1]).to(beat_out.device)
  span_mat[batch_indices, note_indices, beat_indices] = 1
  span_mat[:, :, -1] = 0 # last beat is not used
  spanned_beat = torch.bmm(span_mat, beat_out)
  return spanned_beat

def cal_length_from_padded_beat_numbers(beat_numbers):
  '''
  beat_numbers (torch.Tensor): N x T, zero padded note_location_number

  output (torch.Tensor): N
  '''
  try:
    len_note = torch.min(torch.diff(beat_numbers,dim=1), dim=1)[1] + 1
  except:
    print("Error in cal_length_from_padded_beat_numbers:")
    print(beat_numbers)
    print(beat_numbers.shape)
    [print(beat_n) for beat_n in beat_numbers]
    print(torch.diff(beat_numbers,dim=1))
    print(torch.diff(beat_numbers,dim=1).shape)
    len_note = torch.LongTensor([beat_numbers.shape[1] * len(beat_numbers)]).to(beat_numbers.device)
  len_note[len_note==1] = beat_numbers.shape[1]

  return len_note
