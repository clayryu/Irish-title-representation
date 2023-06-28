import torch
from torch.linalg import norm
from torch.nn import CosineEmbeddingLoss
import torch.nn.functional as F
import time

class ContrastiveLoss():
  def __init__(self, margin=0.4):
    self.margin = margin
    self.non_diag_index_dict = {}

  def __call__(self, emb1, emb2):
    return self.get_batch_contrastive_loss(emb1, emb2)

  def calculate_non_diag_index(self, num_batch):
    if num_batch not in self.non_diag_index_dict:
      self.non_diag_index_dict[num_batch] = [x for x in range(num_batch) for y in range(num_batch) if x!=y], [y for x in range(num_batch) for y in range(num_batch) if x!=y]
    return self.non_diag_index_dict[num_batch]

  def get_batch_contrastive_loss(self, emb1, emb2):
    num_batch = len(emb1)
    dot_product_value = torch.matmul(emb1, emb2.T)
    emb1_norm = norm(emb1, dim=-1)
    emb2_norm = norm(emb2, dim=-1)

    cos_sim_value = dot_product_value / emb1_norm.unsqueeze(1) / emb2_norm.unsqueeze(0)
    positive_sim = cos_sim_value.diag().unsqueeze(1) # N x 1 
    non_diag_index = self.calculate_non_diag_index(num_batch)
    # tuple of two lists, each list has len = N*(N-1)
    # 512 * 511
    negative_sim = cos_sim_value[non_diag_index].reshape(num_batch, num_batch-1)

    loss  = torch.clamp(self.margin - (positive_sim - negative_sim), min=0)
    return loss.mean()

class ContrastiveLoss_euclidean():
  def __init__(self, margin=0.4):
    self.margin = margin
    self.non_diag_index_dict = {}

  def __call__(self, emb1, emb2):
    return self.get_batch_contrastive_loss(emb1, emb2)

  def calculate_non_diag_index(self, num_batch):
    if num_batch not in self.non_diag_index_dict:
      self.non_diag_index_dict[num_batch] = [x for x in range(num_batch) for y in range(num_batch) if x!=y], [y for x in range(num_batch) for y in range(num_batch) if x!=y]
    return self.non_diag_index_dict[num_batch]

  def get_batch_contrastive_loss(self, emb1, emb2):
    num_batch = len(emb1)
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    euclidean_distance = torch.norm(emb1[:, None] - emb2, p=2, dim=-1)

    positive_distance = euclidean_distance.diag().unsqueeze(1) # N x 1 
    non_diag_index = self.calculate_non_diag_index(num_batch)
    # tuple of two lists, each list has len = N*(N-1)
    # 512 * 511
    negative_distance = euclidean_distance[non_diag_index].reshape(num_batch, num_batch-1)

    loss  = torch.clamp(self.margin - (positive_distance - negative_distance), min=0)
    return loss.mean()

def get_batch_euclidean_loss(emb1, emb2, margin=0.4):
  num_batch = len(emb1)
  euclidean_distance = torch.norm(emb1[:, None] - emb2, p=2, dim=-1) # calculate the euclidean distance between the two embeddings
  positive_distance = euclidean_distance.diag().unsqueeze(1) # N x 1 
  non_diag_index = [x for x in range(num_batch) for y in range(num_batch) if x!=y], [y for x in range(len(euclidean_distance)) for y in range(len(euclidean_distance)) if x!=y]
  # tuple of two lists, each list has len = N*(N-1)
  negative_distance = euclidean_distance[non_diag_index].reshape(num_batch, num_batch-1)

  loss  = torch.clamp(margin - (positive_distance - negative_distance), min=0)
  return loss.mean()

def clip_crossentropy_loss(emb1, emb2, tau):
  num_batch = len(emb1)
  dot_product_value = torch.matmul(emb1, emb2.T)
  emb1_norm = norm(emb1, dim=-1)
  emb2_norm = norm(emb2, dim=-1)

  cos_sim_value = dot_product_value / emb1_norm.unsqueeze(1) / emb2_norm.unsqueeze(0) * torch.exp(tau)
  
  n = emb1.shape[0]
  labels = torch.arange(n, dtype=torch.long).to('cuda')
  loss_i = torch.nn.CrossEntropyLoss(reduction='none')(cos_sim_value, labels)
  loss_t = torch.nn.CrossEntropyLoss(reduction='none')(cos_sim_value.T, labels)
  return (loss_i + loss_t).mean()