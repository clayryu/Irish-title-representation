import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import wandb
from wandb import Html
import numpy as np
from torch import dot
from torch.linalg import norm
from torch.nn import CosineEmbeddingLoss

import time
import datetime
import os
import csv
import pickle
from pathlib import Path
from collections import Counter

import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

#from data_utils import decode_melody
from ttl_loss import ContrastiveLoss, ContrastiveLoss_euclidean, clip_crossentropy_loss

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

class EmbTrainer:
  def __init__(self, abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config=None, save_dir=None):
    self.config = config
    self.abc_model = abc_model
    self.ttl_model = ttl_model
    self.abc_optimizer = abc_optimizer
    self.ttl_optimizer = ttl_optimizer
    self.tau = torch.nn.Parameter(torch.tensor(1.0))
    # self.abc_optimizer = torch.optim.Adam(list(abc_model.parameters()) + [self.tau], lr=config.lr)
    # self.ttl_optimizer = torch.optim.Adam(list(ttl_model.parameters()) + [self.tau], lr=config.lr)
    # if config.lr_scheduler_type == 'Plateau':
    #   self.scheduler_abc = torch.optim.lr_scheduler.ReduceLROnPlateau(self.abc_optimizer, 'min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=True)
    #   self.scheduler_ttl = torch.optim.lr_scheduler.ReduceLROnPlateau(self.ttl_optimizer, 'min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=True)
    # elif config.lr_scheduler_type == 'Step':
    #   self.scheduler_abc = torch.optim.lr_scheduler.StepLR(self.abc_optimizer, step_size=config.scheduler_patience, gamma=0.5)
    #   self.scheduler_ttl = torch.optim.lr_scheduler.StepLR(self.ttl_optimizer, step_size=config.scheduler_patience, gamma=0.5)
    
    self.margin = config.train_params.loss_margin
    self.loss_fn = loss_fn
    self.topk = config.train_params.topk

    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.device = config.train_params.device
    self.abc_model.to(self.device)
    self.ttl_model.to(self.device)
    self.tau.to(self.device)
    
    self.grad_clip = config.train_params.grad_clip
    self.best_valid_loss = 100
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    
    self.abc_model_name = config.nn_params.abc_model_name
    self.ttl_model_name = config.nn_params.ttl_model_name
    
    self.num_iter_per_valid = config.train_params.num_valid_per_step
    self.num_epoch_per_log = config.train_params.num_log_per_step
    self.make_log = config.general.make_log
    self.save_dir = save_dir
    self.save_dir_abc = Path(save_dir) / 'abc_model'
    self.save_dir_ttl = Path(save_dir) / 'ttl_model'
    self.save_dir_csv = Path(save_dir) / 'csv'
    self.save_dir_abc.mkdir(exist_ok=True, parents=True)
    self.save_dir_ttl.mkdir(exist_ok=True, parents=True)
    self.save_dir_csv.mkdir(exist_ok=True, parents=True)

    if config.general.query_type == 'title': 
      self.query_type = "Query=T" 
    elif config.general.query_type == 'melody':
      self.query_type = "Query=M"
    
    self.position_tracker = defaultdict(list) # to accumulate the position of the query in the ranking for all the training data

  def save_abc_model(self, path):
    torch.save({'model':self.abc_model.state_dict(), 'optim':self.abc_optimizer.state_dict()}, path)
  
  def save_ttl_model(self, path):
    torch.save({'model':self.ttl_model.state_dict(), 'optim':self.ttl_optimizer.state_dict()}, path)

  def save_dict_to_csv(self, dict_to_save, filename):
    # Specify the filename and mode for writing to the CSV file
    mode = "w"
    sorted_dict = dict(sorted(dict_to_save.items(), key=lambda x: x[1][-1]))

    # Open the file for writing and create a CSV writer object
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)

        # Write the header row with the column names
        writer.writerow(["Key", "Value"])

        # Write each key-value pair to a new row in the CSV file
        for key, value in sorted_dict.items():
            writer.writerow([key, value])
  
  def train_by_num_iter(self, num_iters):
    generator = iter(self.train_loader)
    mrr_table = wandb.Table(columns=["genre", "mrr_value", "step"]) # for logging wandb table
    dcg_table = wandb.Table(columns=["genre", "dcg_value", "step"]) # for logging wandb table
    for i in tqdm(range(num_iters)):
      try:
          # Samples the batch
          batch = next(generator)
      except StopIteration:
          # restart the generator if the previous generator is exhausted.
          generator = iter(self.train_loader)
          batch = next(generator)

      loss_value, loss_dict = self._train_by_single_batch(batch)
      loss_dict = self._rename_dict(loss_dict, 'train')
      if self.make_log:
        wandb.log(loss_dict, step=i)
      self.training_loss.append(loss_value)
      if (i) % self.num_iter_per_valid == 0:
        self.abc_model.eval()
        self.ttl_model.eval()
        validation_metric_dict, genre_mrr_dict, genre_dcg_dict = self.validate(iter=i)
        # if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #   self.scheduler.step(validation_loss)
        validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
        genre_mrr_dict = {k:v for (k,v) in sorted(genre_mrr_dict.items(), key=lambda x: x[0])} # sort by genre name
        genre_dcg_dict = {k:v for (k,v) in sorted(genre_dcg_dict.items(), key=lambda x: x[0])}
        if self.make_log:
          # log mrr values of each genre and fulfill the wandb table
          for j in range(len(genre_mrr_dict)):
            mrr_table.add_data(list(genre_mrr_dict.keys())[j], list(genre_mrr_dict.values())[j], wandb.run.step)
            dcg_table.add_data(list(genre_dcg_dict.keys())[j], list(genre_dcg_dict.values())[j], wandb.run.step)
          #wandb.log(genre_mrr_dict, step=i)
          wandb.log(genre_dcg_dict, step=i)
          wandb.log(validation_metric_dict, step=i)

        self.save_dict_to_csv(self.position_tracker, f"{self.save_dir_csv}/{self.query_type}_position_tracker_{i}.csv")
        self.save_dict_to_csv(self.answer_tracker, f"{self.save_dir_csv}/{self.query_type}_answer_tracker_{i}.csv")
        self.save_dict_to_csv(self.metric_by_genre, f"{self.save_dir_csv}/{self.query_type}_metric_by_genre_{i}.csv")

        self.validation_loss.append(validation_metric_dict['valid.validation_loss'])
        self.validation_acc.append(validation_metric_dict['valid.validation_MRR'])
        validation_loss = validation_metric_dict['valid.validation_loss']
        self.best_valid_loss = min(validation_metric_dict['valid.validation_loss'], self.best_valid_loss)
        
        if (i) % (self.num_epoch_per_log * self.num_iter_per_valid) == 0:
          # self.inference_and_log(i) # TODO: implement this
          self.save_abc_model(self.save_dir_abc / f'iter{i}_loss{validation_loss:.4f}.pt')
          self.save_ttl_model(self.save_dir_ttl / f'iter{i}_loss{validation_loss:.4f}.pt')
        self.abc_model.train()
        self.ttl_model.train()

    wandb.log({"multiline": wandb.plot_table( 
        "wandb/line/v0", mrr_table, {"x": "step", "y": "mrr_value", "groupKeys": "genre"}, {"title": "mrr_by_genre"})
    })

    wandb.finish()    

  def train_by_num_epoch(self, num_epochs):
    date = datetime.datetime.now().strftime("%Y%m%d") # for saving model
    for epoch in tqdm(range(num_epochs)):
      print(self.abc_optimizer.param_groups[0]['lr'])
      self.abc_model.train()
      self.ttl_model.train()
      start_time = time.time()
      for batch in self.train_loader:
        print(f"batch_time: {time.time() - start_time}")
        loss_value, loss_dict = self._train_by_single_batch(batch, reg_abc_loss=None, reg_ttl_loss=None)
        print('passed train_by_single_batch')
        loss_dict = self._rename_dict(loss_dict, 'train')
        # if self.make_log:
        #   wandb.log(loss_dict)
        self.training_loss.append(loss_value)
      
      epoch_time = time.time() - start_time
      if self.make_log:
        wandb.log({
                  # "validation_loss": validation_loss,
                  # "validation_acc": validation_acc,
                  "train_loss": loss_value,
                  # "train_acc": train_acc,
                  "train.time": epoch_time
                  })

      if epoch % 40 == 0:
        self.abc_model.eval()
        self.ttl_model.eval()
        validation_dict_train = self.validate(external_loader=self.train_loader, epoch=epoch)
        validation_dict_valid = self.validate(epoch=epoch)
        if isinstance(self.scheduler_abc, torch.optim.lr_scheduler.ReduceLROnPlateau):
          self.scheduler_abc.step(validation_dict_valid['validation_loss'])
          self.scheduler_ttl.step(validation_dict_valid['validation_loss'])
        #self.validation_loss.append(validation_loss)
        #self.validation_acc.append(validation_acc)
        if not os.path.exists(f'saved_models/{date}/{self.config.name_of_model_to_save}'):
          os.makedirs(f'saved_models/{date}/{self.config.name_of_model_to_save}')
        self.save_dict_to_csv(self.position_tracker, f"saved_models/{date}/{self.config.name_of_model_to_save}/{self.config.name_of_model_to_save}_position_tracker_{epoch}.csv")
        
        if self.make_log:
          wandb.log(validation_dict_valid)
          wandb.log({
                    "train_acc": validation_dict_train['validation_acc'],
          })

        self.best_valid_loss = min(validation_dict_valid['validation_loss'], self.best_valid_loss)
        if epoch % 250 == 0:
          # if directory does not exist, make directory
          self.save_abc_model(f'saved_models/{date}/{self.config.name_of_model_to_save}/{self.abc_model_name}_{epoch}.pt')
          self.save_ttl_model(f'saved_models/{date}/{self.config.name_of_model_to_save}/{self.ttl_model_name}_{epoch}.pt')
      
  def _train_by_single_batch(self, batch, reg_abc_loss=None, reg_ttl_loss=None):

    # start_time = time.time()
    loss, loss_dict = self.get_loss_pred_from_single_batch(batch)
    if reg_abc_loss is not None:
      loss = loss + reg_abc_loss + reg_ttl_loss
                          
    if loss != 0:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(list(self.abc_model.parameters()) + [self.tau], self.grad_clip)
      torch.nn.utils.clip_grad_norm_(list(self.ttl_model.parameters()) + [self.tau], self.grad_clip)
                        
      self.abc_optimizer.step()
      self.abc_optimizer.zero_grad()

      self.ttl_optimizer.step()
      self.ttl_optimizer.zero_grad()
      
      # if not isinstance(self.scheduler_abc, torch.optim.lr_scheduler.ReduceLROnPlateau):
      #   self.scheduler_abc.step()
      #   self.scheduler_ttl.step()
    
    return loss.item(), loss_dict
  
  def get_loss_pred_from_single_batch(self, batch):
    melody, title = batch
    # check time
    emb1 = self.abc_model(melody.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    if self.loss_fn == CosineEmbeddingLoss():
      loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
    elif self.loss_fn == get_batch_contrastive_loss:
      start_time = time.time()
      loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
      print(f"get_batch_contrastive_loss time: {time.time() - start_time}")
    elif self.loss_fn == get_batch_euclidean_loss:
      loss = get_batch_euclidean_loss(emb1, emb2)
    elif self.loss_fn == clip_crossentropy_loss:
      loss = clip_crossentropy_loss(emb1, emb2, self.tau)
      
    loss_dict = {'total': loss.item()}
    return loss, loss_dict
    
  def validate(self, external_loader=None, topk=20):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
      
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
    '''
    
    ### Don't change this part
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.abc_model.eval()
    self.ttl_model.eval()
                          
    '''
    Write your code from here, using loader, self.model, self.loss_fn_fn.
    '''
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    correct_emb = 0
    
    abc_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title = batch
        
        emb1 = self.abc_model(melody.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1:
            continue
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        elif self.loss_fn == get_batch_contrastive_loss:
          loss = get_batch_contrastive_loss(emb1, emb2, self.margin)
        elif self.loss_fn == get_batch_euclidean_loss:
          loss = get_batch_euclidean_loss(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
        
        '''
        cos_emb1 = emb1.detach().cpu().numpy()
        cos_emb2 = emb2.detach().cpu().numpy()
        # sklearn 사용을 위해서 numpy로 바꿔주고 다시 tensor로 바꿔 대각 요소들을 빼내어 1대1 비교값의 리스트를 만든다.
        cos_emb = torch.tensor(cosine_similarity(cos_emb1, cos_emb2)).diag() 
        acc = torch.sum(cos_emb > 0.5)

        validation_acc += acc.item()
        total_sentence += len(melody[3]) # packed sequence의 3번째 리스트는 배치된 문장의 순서이다.
        '''
      if self.loss_fn == get_batch_contrastive_loss or self.loss_fn == CosineEmbeddingLoss:
        cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy())
        sorted_cos_idx = np.argsort(cos_sim, axis=-1)
        for idx in range(len(loader.dataset)):
            if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
                correct_emb += 1
                
      elif self.loss_fn == get_batch_euclidean_loss:
        euc_sim = torch.norm(abc_emb_all[:, None] - emb2, p=2, dim=-1)
        sorted_euc_idx = np.argsort(euc_sim, axis=-1)
        for idx in range(len(loader.dataset)):
            if idx in sorted_euc_idx[idx][-topk:]: # pick best 20 scores
                correct_emb += 1
        
    return validation_loss / num_total_tokens, correct_emb / len(self.valid_loader.dataset)
  
  def _rename_dict(self, adict, prefix='train'):
    keys = list(adict.keys())
    for key in keys:
      adict[f'{prefix}.{key}'] = adict.pop(key)
    return dict(adict)
  
class EmbTrainerMeasure(EmbTrainer):
  def __init__(self, abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config=None, save_dir=None):
    super().__init__(abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config, save_dir)
    
  def get_loss_pred_from_single_batch(self, batch):
    melody, title, measure_numbers, ttltext = batch
    if self.config.general.input_feat == "all" or self.config.general.input_feat=="all_except_genre" or self.config.general.input_feat=='melody_only': # 4feat + melody
      emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
    elif self.config.general.input_feat == "header":
      emb1 = self.abc_model(melody.to(self.device))
    emb2 = self.ttl_model(title.to(self.device))
    
    if self.loss_fn == CosineEmbeddingLoss():
      loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
    elif isinstance(self.loss_fn, ContrastiveLoss):
      start_time = time.time()
      loss = self.loss_fn(emb1, emb2)
      # print(f"get_batch_contrastive_loss time: {time.time() - start_time}")
    elif isinstance(self.loss_fn, ContrastiveLoss_euclidean):
      loss = self.loss_fn(emb1, emb2)
    elif self.loss_fn == clip_crossentropy_loss:
      loss = clip_crossentropy_loss(emb1, emb2, self.tau)
      
    loss_dict = {'total': loss.item()}
    return loss, loss_dict
  
  def validate(self, external_loader=None, topk=20):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.abc_model.eval()
    self.ttl_model.eval()
                          
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    total_sentence = 0
    correct_emb = 0
    
    abc_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.emb_size)

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title, measure_numbers = batch
        
        emb1 = self.abc_model(melody.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(title)

        abc_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        
        if len(melody[3]) == 1: # got 1 batch
            continue
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        elif isinstance(self.loss_fn, ContrastiveLoss):
          loss = self.loss_fn(emb1, emb2)
        elif isinstance(self.loss_fn, ContrastiveLoss_euclidean):
          loss = self.loss_fn(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
      '''
      # for comparing MRR loss between randomly initialized model and trained model
      abc_emb_all = torch.rand(len(self.valid_loader.dataset), 128) # valid dataset size(10% of all) x embedding size 128
      ttl_emb_all = torch.rand(len(self.valid_loader.dataset), 128)
      '''
      cos_sim = cosine_similarity(abc_emb_all.detach().cpu().numpy(), ttl_emb_all.detach().cpu().numpy()) # a x b mat b x a = a x a
      sorted_cos_idx = np.argsort(cos_sim, axis=-1)
      for idx in range(len(loader.dataset)):
        if idx in sorted_cos_idx[idx][-topk:]: # pick best 20 scores
          correct_emb += 1

        
    return validation_loss / num_total_tokens, correct_emb / len(self.valid_loader.dataset)
  
class EmbTrainerMeasureMRR(EmbTrainerMeasure):
  def __init__(self, abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config=None, save_dir=None):
    super().__init__(abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config, save_dir)
  
  def validate(self, external_loader=None, iter=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.abc_model.eval()
    self.ttl_model.eval()
                          
    validation_loss = 0
    num_total_tokens = 0
    sum_mrr = 0
    sum_dcg = 0
    self.answer_tracker = defaultdict(list)
    self.metric_by_genre = defaultdict(list)
    self.total_genre_mrr_score = defaultdict(float)
    self.total_genre_dcg_score = defaultdict(float)

    # dict of title and its genre
    with open('/home/clay/userdata/title_representation/title_genre_dict.pickle', 'rb') as f:
      title_genre_dict = pickle.load(f)
    genre_count = Counter(title_genre_dict.values())

    melody_emb_all = torch.zeros(len(loader.dataset), self.abc_model.output_size) # valid dataset size(10% of all) x embedding size
    ttl_emb_all = torch.zeros(len(loader.dataset), self.abc_model.output_size)
    ttl_text_all = []

    with torch.inference_mode():
      for idx, batch in enumerate(tqdm(loader, leave=False)):
        melody, title, measure_numbers, ttltext = batch
        if self.config.general.input_feat == "all" or self.config.general.input_feat=="all_except_genre" or self.config.general.input_feat=='melody_only': # 4feat + melody
          emb1 = self.abc_model(melody.to(self.device), measure_numbers.to(self.device))
        elif self.config.general.input_feat == "header":
          emb1 = self.abc_model(melody.to(self.device))
        emb2 = self.ttl_model(title.to(self.device))

        start_idx = idx * loader.batch_size
        end_idx = start_idx + len(emb1)

        #emb1 = F.normalize(emb1, p=2, dim=-1)
        #emb2 = F.normalize(emb2, p=2, dim=-1)

        melody_emb_all[start_idx:end_idx] = emb1
        ttl_emb_all[start_idx:end_idx] = emb2
        if loader == self.valid_loader:
          ttl_text_all.extend(ttltext)
        
        if len(melody[3]) == 1: # got 1 batch
            continue
        
        if self.loss_fn == CosineEmbeddingLoss():
          loss = self.loss_fn(emb1, emb2, torch.ones(emb1.size(0)).to(self.device))
        elif isinstance(self.loss_fn, ContrastiveLoss):
          loss = self.loss_fn(emb1, emb2)
        elif isinstance(self.loss_fn, ContrastiveLoss_euclidean):
          loss = self.loss_fn(emb1, emb2)
        elif self.loss_fn == clip_crossentropy_loss:
          loss = clip_crossentropy_loss(emb1, emb2, self.tau)
        
        num_tokens = melody.data.shape[0] # number of tokens ex) torch.Size([5374, 20])
        validation_loss += loss.item() * num_tokens
        #print(validation_loss)
        num_total_tokens += num_tokens
      '''
      # for comparing MRR loss between randomly initialized model and trained model
      abc_emb_all = torch.rand(len(self.valid_loader.dataset), 128) # valid dataset size(10% of all) x embedding size 128
      ttl_emb_all = torch.rand(len(self.valid_loader.dataset), 128)
      '''
      
      if self.topk != 0:
        topk = self.topk
      else:
        topk = len(loader.dataset)

      if isinstance(self.loss_fn, ContrastiveLoss) or self.loss_fn == CosineEmbeddingLoss or self.loss_fn == clip_crossentropy_loss:
        if loader == self.valid_loader:
          torch.save(melody_emb_all, f'{self.save_dir_abc}/abc_emb_{iter}.pt')
          torch.save(ttl_emb_all, f'{self.save_dir_ttl}/ttl_emb_{iter}.pt')

        if self.query_type == "Query=T":
          row_emb = ttl_emb_all
          column_emb = melody_emb_all
        elif self.query_type == "Query=M":
          row_emb = melody_emb_all
          column_emb = ttl_emb_all

        dot_product_value = torch.matmul(row_emb, column_emb.T)
        row_emb_norm = norm(row_emb, dim=-1)
        column_emb_norm = norm(column_emb, dim=-1)

        cos_sim_value = dot_product_value / row_emb_norm.unsqueeze(1) / column_emb_norm.unsqueeze(0)
        cos_sim = cos_sim_value.detach().cpu().numpy()
        sorted_cos_idx = np.argsort(cos_sim, axis=-1) # the most similar one goes to the end
        # calculate MRR
        mrrdict = {i-1:1/i for i in range(1, topk+1)} # {0:1.0, 1:0.5, ...}
        # calculate DCG
        dcg_dict = {i-1: (2**1 - 1) / np.log2(i + 1) for i in range(1, topk+1)} # {0:1.0, 1:0.6309297535714574, ...}

        for idx in range(len(loader.dataset)):
          if idx in sorted_cos_idx[idx][-topk:]: # pick topk if topk parameter is 0 then choose all
            position = np.argwhere(sorted_cos_idx[idx][-topk:][::-1] == idx).item() # changing into ascending order
            quality_score = mrrdict[position]
            sum_mrr += quality_score
            relevance_score = 1 # you can replace 1 with your own relevance score
            dcg = relevance_score * dcg_dict[position]
            sum_dcg += dcg
          if loader == self.valid_loader:
            self.position_tracker[ttl_text_all[idx]].append(position)
            title_name = [ttl_text_all[i] for i in sorted_cos_idx[idx][-10:][::-1]] # top 10 titles
            title_name.append(position) # using position values to sort two csvs in the same order
            self.answer_tracker[ttl_text_all[idx]] = title_name

            # Dict of Metric values by each genre
            genre_name = [title_genre_dict[ttl_text_all[i]] for i in sorted_cos_idx[idx][-topk:][::-1]]
            genre_mrr_score_dict, genre_dcg_score_dict = self.calculate_metric_by_genre(genre_name, mrrdict, dcg_dict, genre_count)
            self.metric_by_genre[ttl_text_all[idx]].append(title_genre_dict[ttl_text_all[idx]])
            self.metric_by_genre[ttl_text_all[idx]].append(genre_mrr_score_dict)
            self.metric_by_genre[ttl_text_all[idx]].append(genre_dcg_score_dict)
            self.metric_by_genre[ttl_text_all[idx]].append(position)

        # calculate nDCG
        # ideal DCG must be 1.0 in the case of perfect ranking in the case when there only one correct answer
        # when [correct, pred, pred, pred, ...]
        # so sum_dcg can be considered as nDCG
                
      with open(f'{self.save_dir}/ttl_text_all.pkl', 'wb') as f:
        pickle.dump(ttl_text_all, f)

      validation_dict = {'validation_loss': validation_loss / num_total_tokens, 'validation_MRR': sum_mrr / len(loader.dataset), 'validation_DCG': sum_dcg / len(loader.dataset)}
      self.total_genre_mrr_score = {k: v / len(loader.dataset) for k, v in self.total_genre_mrr_score.items()} # average of all songs
      self.total_genre_dcg_score = {k: v / len(loader.dataset) for k, v in self.total_genre_dcg_score.items()} # average of all songs

    return validation_dict, self.total_genre_mrr_score, self.total_genre_dcg_score

  def calculate_metric_by_genre(self, genre_name, mrr_dict, dcg_dict, genre_count):
    genre_mrr_score = defaultdict(float)
    genre_dcg_score = defaultdict(float)
    for idx, genre in enumerate(genre_name):
      mrr_score = mrr_dict[idx]
      mrr_score = mrr_score / genre_count[genre]
      dcg_score = dcg_dict[idx]
      dcg_score = dcg_score / genre_count[genre]
      genre_mrr_score[genre] += mrr_score
      self.total_genre_mrr_score[genre] += mrr_score
      genre_dcg_score[genre] += dcg_score
      self.total_genre_dcg_score[genre] += dcg_score

    genre_mrr_score = {k: v for k, v in sorted(list(genre_mrr_score.items()), key=lambda x: x[1], reverse=True)}
    genre_dcg_score = {k: v for k, v in sorted(list(genre_dcg_score.items()), key=lambda x: x[1], reverse=True)}
    return genre_mrr_score, genre_dcg_score