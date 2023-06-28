import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ttl_model_zoo
import ttl_loss
import ttl_utils
from ttl_trainer import EmbTrainerMeasureMRR
from ttl_data_utils import read_tunes, prepare_abc, convert_token, is_valid_tune, Dataset_4feat_title

from pathlib import Path
from pyabc import pyabc
from tqdm.auto import tqdm

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def check_tune_validity(tune):
  if len(tune.measures) == 0 or tune.measures[-1].number == 0:
    return False
  if not tune.is_ending_with_bar:
    return False
  if tune.is_tune_with_full_measures or tune.is_incomplete_only_one_measure:
    if is_valid_tune(tune):
      return True
  else:
    return False
  
def make_input(tune_list):
  unique_title_list = []
  unique_tune_list = []
  genre_list = []
  key_list = []
  meter_list = []
  unit_note_length_list = []
  for tune in tune_list:
    if tune.title.split(' ')[-1] == 'The':
      list_title = tune.title.split(' ')[:-1]
      list_title.insert(0, 'The')
      new_title = ' '.join(list_title)[:-1]
    else:
      new_title = tune.title
    if new_title == 'x' or new_title == 'Untitled':
      continue
    if new_title not in unique_title_list:
      unique_title_list.append(new_title)
      unique_tune_list.append(tune)
      genre_list.append(tune.header['rhythm'])
      try:
        key_list.append(tune.header['transcription'])
      except:
        key_list.append(tune.header['key'])
      meter_list.append(tune.header['meter'])
      unit_note_length_list.append(tune.header['unit note length'])
  return unique_title_list, unique_tune_list, genre_list, key_list, meter_list, unit_note_length_list

@hydra.main(version_base=None, config_path="./yamls/", config_name="config")
def main(config: DictConfig):
  if config.general.make_log:
    wandb.init(
      # set the wandb project where this run will be logged
      project="title_representation",
      # set the entity
      entity="clayryu",
      # track hyperparameters and run metadata
      config = OmegaConf.to_container(config)
    )
    save_dir = wandb.run.dir + '/checkpoints/'
  else: 
    save_dir = 'wandb/debug/checkpoints/'

  score_dir = Path('abc_dataset/folk_rnn_abc_key_cleaned/')
  abc_list = list(score_dir.rglob('*.abc')) + list(score_dir.rglob('*.ABC'))
  tune_list, _ = prepare_abc(abc_list)
  tune_list = [tune for tune in tune_list if check_tune_validity(tune)]
  unique_title_list, unique_tune_list, genre_list, key_list, meter_list, unit_note_length_list = make_input(tune_list)

  dataset = Dataset_4feat_title(unique_title_list, genre_list, key_list, meter_list, unit_note_length_list)

  trainset, validset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(42))
  train_loader = DataLoader(trainset, batch_size=3000, shuffle=True, num_workers=0)
  valid_loader = DataLoader(validset, batch_size=3000, shuffle=False, num_workers=0)

  ttl_model = ttl_model_zoo.TTL_Emb_Model_noconfig()
  #header_model = ttl_model_zoo.Header_Emb_model(dataset.get_vocab())
  header_model = ttl_model_zoo.Header_Embedding_Model_forGK(dataset.get_vocab())
  #header_model = ttl_model_zoo.Header_Embedding_Model_forMU(dataset.get_vocab())

  ttl_optimizer = torch.optim.Adam(ttl_model.parameters(), lr=config.train_params.lr)
  header_optimizer = torch.optim.Adam(header_model.parameters(), lr=config.train_params.lr)

  loss_fn = ttl_loss.ContrastiveLoss(margin=config.train_params.loss_margin)

  trainer = EmbTrainerMeasureMRR(header_model, ttl_model, header_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config, save_dir)
  trainer.train_by_num_iter(config.train_params.num_iter)

if __name__ == "__main__":
  main()