import torch
from torch.utils.data import DataLoader

from ttl_loss import ContrastiveLoss
from ttl_utils import pack_collate_title_sampling_textttl, Pack_collate_sampling
from ttl_trainer import EmbTrainerMeasureMRR
import pre_vocab_utils
import ttl_data_utils
import ttl_model_zoo
import ttl_data_utils
import vocab_utils
import pre_model_zoo

import argparse
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


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

  print(config.nn_params.abc_model_name, config.nn_params.ttl_model_name)

  # setting vocab with pre-trained word embedding
  vocab_path = config.data_params.vocab_path
  score_path = config.data_params.score_path
  pre_params = config.pre_trained.nn_params
  pre_params_resize = config.pre_trained_resize.nn_params # reduce layer numbers in rnn for gpu memory
  net_params = config.nn_params
  vocab = getattr(pre_vocab_utils, pre_params.vocab_name)(json_path = vocab_path)
  pre_params = ttl_data_utils.get_emb_total_size(pre_params, vocab)
  pre_params_resize = ttl_data_utils.get_emb_total_size(pre_params_resize, vocab)
  if config.general.debug:
    dataset_abc = getattr(ttl_data_utils, config.data_params.dataset)(score_path, vocab_path, num_limit=70, vocab_name=pre_params.vocab_name, config=config, pre_vocab=vocab)
  else:
    dataset_abc = getattr(ttl_data_utils, config.data_params.dataset)(score_path, vocab_path, vocab_name=pre_params.vocab_name, config=config, pre_vocab=vocab)

  # setting abc embedding model

  abc_model_name = net_params.abc_model_name
  if 'CNN' in abc_model_name:
    abc_model = getattr(ttl_model_zoo, abc_model_name)(trans_emb=None, vocab_size=vocab.get_size(), net_param=net_params, pre_param=pre_params, emb_ratio=1) # print conv layer in model
  elif 'RNN' in abc_model_name:
    # load pre-trained model
    checkpoint_path = config.data_params.checkpoint_path
    if config.general.using_pretrained:
      rnn_gen_model = getattr(pre_model_zoo, pre_params.model_name)(vocab.get_size(), pre_params)
      checkpoint = torch.load(checkpoint_path, map_location= 'cpu')
      rnn_gen_model.load_state_dict(checkpoint['model'])
      # freeze pre-trained model
      for param in rnn_gen_model.parameters():
        param.requires_grad = False
    else: # reduce num_layers of each rnn layer for gpu memory
      rnn_gen_model = getattr(pre_model_zoo, pre_params_resize.model_name)(vocab.get_size(), pre_params_resize)

    # load model
    abc_model = getattr(ttl_model_zoo, abc_model_name)(rnn_gen_model.emb, rnn_gen_model.rnn, rnn_gen_model.measure_rnn, rnn_gen_model.final_rnn, net_param=net_params)
    # abc_model = ttl_model_zoo.Melody_Only_Model()
    
  # setting title embedding model
  ttl_model_name = net_params.ttl_model_name
  ttl_in_embedding_size = 768 if config.text_embd_params.model_type[:2] == 'ST' else 1536
  ttl_model = getattr(ttl_model_zoo, ttl_model_name)(in_embedding_size=ttl_in_embedding_size, net_param=net_params)

  # setting optimizer and loss function
  abc_optimizer = torch.optim.Adam(abc_model.parameters(), lr=config.train_params.lr)
  ttl_optimizer = torch.optim.Adam(ttl_model.parameters(), lr=config.train_params.lr)
  loss_fn = ContrastiveLoss(margin=config.train_params.loss_margin)

  # setting trainer
  trainset, validset = torch.utils.data.random_split(dataset_abc, [int(len(dataset_abc)*0.9), len(dataset_abc) - int(len(dataset_abc)*0.9)], generator=torch.Generator().manual_seed(42))
  train_loader = DataLoader(trainset, batch_size=config.train_params.batch_size, collate_fn=Pack_collate_sampling(sample_num=config.train_params.num_sampling), shuffle=True, num_workers=0)
  valid_loader = DataLoader(validset, batch_size=config.train_params.batch_size, collate_fn=Pack_collate_sampling(sample_num=config.train_params.num_sampling), shuffle=False, num_workers=0)

  trainer = EmbTrainerMeasureMRR(abc_model, ttl_model, abc_optimizer, ttl_optimizer, loss_fn, train_loader, valid_loader, config, save_dir)
  trainer.train_by_num_iter(config.train_params.num_iter)

if __name__ == "__main__":
  main()
