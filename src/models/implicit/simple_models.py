# Make it possible to run this script from any dir, by adding src and lib folders to the sys path
import sys, os
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent.parent)
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)
LIB_DIR = os.path.join(ROOT_DIR, 'lib')
sys.path.append(LIB_DIR)

import traceback
import itertools
import argparse
import copy
from doctest import script_from_examples
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloaders import *
from utils import *
from modules.ff import FFNet
from modules.gnn.mpgnn import TemporalEmbeddedMPGNN
from modules.combined import TransformerFF, GRUFF, ProjectTimeFFNet, ProjectFeaturesFFNet
#from modules.combined import TransformerMPGNN

from tqdm import tqdm

from torchinfo import summary

import copy

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

import ray
from ray import tune
from ray.tune import CLIReporter

def iterate_batch(loader, config):
  if config['ray_tune']:
    for x, y in loader:
      yield x, y, None
  else:
    with tqdm(loader, unit="batch") as tepoch:
      for x, y in tepoch:
        yield x, y, tepoch

def grad_to_tb(writer, model, step, prefix=""):
  if prefix:
    prefix = f"{prefix}/" 
  for i, (tag, value) in enumerate(model.named_parameters()):
    if value.grad is not None:
      #print(f"{tag} size: {value.grad.shape}, type: {value.grad.type()}")
      writer.add_histogram(f"{prefix}grad/{i}-{tag}", value.grad, step)


def eval(loaders, model, criterion, config, split, epoch, current_es, writer):
  device = config['device']
  experiment = config['experiment']
  max_epochs = config['max_epochs']
  early_stopping = config['early_stopping']

  losses = []
  ys = []
  pts = []
  with torch.no_grad():
    for (x, y, tepoch) in iterate_batch(loaders[split], config):
      if type(x) == dict:
        for k, v in x.items():
          x[k] = v.to(device)
      else:
        x = x.to(device)
      pt = model(x)
      if config['loss'] == 'mse' and config['scale_output']:
        pt = (pt * (config['category_count']-1)) + 1
      if config['loss'] == 'bce':
        y = y.long()-1
      ys.append(y.numpy())
      # handle single output
      if len(pt.shape) == 1:
        pt = pt.unsqueeze(0)
      pts.append(pt.detach().cpu().numpy())
      loss = criterion(pt, y.to(device))
      if tepoch is not None:
        tepoch.set_description(f"{experiment} {split} Epoch {epoch}/{max_epochs}, loss: {loss}, es: {current_es}/{early_stopping}")
      # total loss
      losses.append(loss.item())
  mean_loss = np.array(losses).mean()
  writer.add_scalar(f'{split}/loss', mean_loss, epoch)
  ys = np.hstack(ys)
  if config['loss'] == 'bce':
    pts = np.concatenate(pts).argmax(axis=1)
  elif config['loss'] == 'mse':
    pts = np.concatenate(pts).round()
  f1 = f1_score(ys, pts, average='macro')
  bin_f1 = f1_score(ys>2, pts>2, average='macro')
  l1 = np.linalg.norm(ys-pts, ord=1)/pts.shape[0]
  acc = accuracy_score(ys, pts)
  writer.add_scalar(f'{split}/f1', f1, epoch)
  writer.add_scalar(f'{split}/bin_f1', bin_f1, epoch)
  writer.add_scalar(f'{split}/l1', l1, epoch)
  writer.add_scalar(f'{split}/acc', acc, epoch)
  return {
    'loss': mean_loss,
    'f1': f1,
    'acc': acc,
    'l1': l1,
  }

def find_best_weights(config, base="/data/ray"):
  experiment = config['experiment']
  params = f"batch_size={config['batch_size']},dropout={config['dropout']:0.4f},learning_rate={config['learning_rate']}"
  searchpath = f"{base}/{experiment}/*/*{params}*/experiments/running/{experiment}/*.pt"
  print(f"Searching for best weights in {searchpath}.")
  results = list(glob.iglob(searchpath))
  print(f"Found {len(results)} results.")
  return results[0]

def load_config(config, with_weights=False, weight_file=None, weight_base="/data/ray"):
  print(f"Config:\n{json.dumps(config, indent=2, default=str, sort_keys=True)}"),

  criterion = None
  if config['loss'] == 'mse':
    criterion = torch.nn.MSELoss()
  elif config['loss'] == 'bce':
    criterion = torch.nn.CrossEntropyLoss()
  assert criterion is not None

  loaders = implicit_pytorch_split_loaders(config)

  # get standard shapes
  shapes = loaders['train'].dataset.shapes

  if config['timesteps'] != shapes['num_timesteps']:
    raise Exception('Config timesteps ({}) does not match data timesteps ({})'.format(config['timesteps'], shapes['num_timesteps']))

  config['num_classes'] = 5
  if config['loss'] == 'mse':
    config['osize'] = 1
  elif config['loss'] == 'bce':
    config['osize'] = config['num_classes']
  else:
    raise Exception('Unknown loss: {}'.format(config['loss']))

  timesteps = shapes['num_timesteps']
  if config['filter_to_timesteps'] > 0:
    timesteps = config['filter_to_timesteps']

  model = None
  if config['module'] == 'ffnet':
    model = FFNet(
      isize=shapes['num_features'],
      osize=config['osize'],
      internal_sizes=config['ffnet_sizes'],
      dropout=config['dropout'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'projecttimeff':
    model = ProjectTimeFFNet(
      input_dim=shapes['num_features'],
      embed_dim=config['temporal_embed_size'],
      timesteps=shapes['num_timesteps'],
      output_dim=config['osize'],
      internal_sizes=config['ffnet_sizes'],
      dropout=config['dropout'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'projectfeaturesff':
    model = ProjectFeaturesFFNet(
      embed_dim=config['feature_embed_size'],
      timesteps=timesteps,
      size_by_feature=shapes['by_feature'],
      output_dim=config['osize'],
      internal_sizes=config['ffnet_sizes'],
      dropout=config['dropout'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'transformer':
    # input dims are (batch, seq_len, input_dim) and output is the same
    model = TransformerFF(
      isize=shapes['num_features'],
      osize=config['osize'],
      timesteps=shapes['num_timesteps'],
      hidden_dim=config['temporal_hidden_dim'],
      embed_dim=config['temporal_embed_size'],
      num_heads=config['transformer_num_heads'],
      num_layers=config['transformer_num_layers'],
      ffnet_sizes=config['ffnet_sizes'],
      with_pos_enc=config['transformer_with_pos_enc'],
      dropout=config['dropout'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'gru':
    model = GRUFF(
      isize=shapes['num_features'],
      osize=config['osize'],
      timesteps=config['timesteps'],
      hidden_dim=config['temporal_hidden_dim'],
      embed_dims=config['temporal_embed_size'],
      num_layers=config['gru_num_layers'],
      ffnet_sizes=config['ffnet_sizes'],
      dropout=config['dropout'],
      bidirectional=config['gru_bidirectional'],
      all_hidden_states=config['gru_all_hidden_states'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'tgnn':
    model = TemporalEmbeddedMPGNN(
      gga_dim=shapes['gga_features'],
      batch_size=config['batch_size'],
      temporal_embed_dim=config['temporal_embed_size'],
      timesteps=timesteps,
      node_features=shapes['node_features'],
      intermediate_node_features=config['gnn_intermediate_node_features'],
      edge_features=shapes['edge_features'],
      intermediate_edge_features=config['gnn_intermediate_edge_features'],
      num_nodes=shapes['num_nodes'],
      osize=config['osize'],
      gnn_internal_sizes=config['gnn_sizes'],
      ffnet_internal_sizes=config['ffnet_sizes'],
      aggregate_over=config['aggregate_gnn_features_over'],
      dropout=config['dropout'],
      verbose=config['verbose'],
      activation=config['activation'],
      last_activation=config['last_activation'],
    ).to(config['device'])
  elif config['module'] == 'transformer_gnn':
    raise RuntimeError("Not implemented")
    ## input dims are (batch, seq_len, input_dim) and output is the same
    #model = TransformerMPGNN(
    #  node_features=shapes['node_features'],
    #  gnn_intermediate_node_features=config['gnn_intermediate_node_features'],
    #  edge_features=shapes['edge_features'],
    #  gnn_intermediate_edge_features=config['gnn_intermediate_edge_features'],
    #  osize=config['osize'],
    #  timesteps=shapes['num_timesteps'],
    #  node_transformer_sizes=dict(
    #    hidden_dim=config['temporal_hidden_dim'],
    #    embed_dim=config['temporal_embed_size'],
    #    intermediate_features=config['transformer_intermediate_node_features'],
    #    num_heads=4,
    #    num_layers=2,
    #  ),
    #  edge_transformer_sizes=dict(
    #    hidden_dim=config['temporal_hidden_dim'],
    #    embed_dim=config['temporal_embed_size'],
    #    intermediate_features=config['transformer_intermediate_edge_features'],
    #    num_heads=2,
    #    num_layers=1,
    #  ),
    #  num_nodes=shapes['num_nodes'],
    #  gnn_internal_sizes=config['gnn_sizes'],
    #  ffnet_sizes=config['ffnet_sizes'],
    #  aggregate_gnn_features_over=config['aggregate_gnn_features_over'],
    #  dropout=config['dropout'],
    #  verbose=config['verbose'],
    #  activation=config['activation'],
    #  last_activation=config['last_activation'],
    #).to(config['device'])
  elif config['module'] == 'temporal_gnn':
    #  model = TemporalEmbeddedMPGNN()
    raise RuntimeError("Not Fully implemented")
  else:
    raise RuntimeError('Unknown module: {}'.format(config['module']))

  # load up weights
  if with_weights:
    if weight_file:
      model.load_state_dict(torch.load(weight_file, map_location=config['device']))
    elif weight_base:
      model.load_state_dict(torch.load(find_best_weights(config, weight_base), map_location=config['device']))
    else:
      raise RuntimeError('No weight_file or weight_base provided')


  return criterion, loaders, shapes, model

def train_loop(config):
  print(f"Starting train loop")
  if config['ray_tune']:
    current_gpu = ray.get_gpu_ids()[0]
    print(f" GPU {current_gpu}")
    #try:
    #  wait_for_gpu(target_util=0.02)
    #except RuntimeError as e:
    #  print(f"GPU {current_gpu} not available: {e}")

  best_val_loss = None
  current_es = 0

  writer = SummaryWriter(config['tensorboard_path'])

  criterion, loaders, shapes, model = load_config(config)

  model.train()

  print(f"Model: {model} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
  summary(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

  device = config['device']
  experiment = config['experiment']
  max_epochs = config['max_epochs']
  early_stopping = config['early_stopping']
  best_model = None

  for epoch in range(max_epochs):
    model.train()
    total_train_loss = []
    ys = []
    pts = []
    for (x, y, tepoch) in iterate_batch(loaders['train'], config):
      with torch.autograd.set_detect_anomaly(mode=config['detect_anomaly'], check_nan=True):
        optimizer.zero_grad()
        # one training step
        if type(x) == dict:
          for k, v in x.items():
            x[k] = v.to(device)
        else:
          x = x.to(device)
        pt = model(x)
        if config['loss'] == 'mse' and config['scale_output']:
          pt = (pt * (config['category_count']-1)) + 1
        ys.append(y.numpy())
        # handle single output
        if not len(pt.shape):
          pt = pt.unsqueeze(0)
        pts.append(pt.detach().cpu().numpy())
        y = y.to(device)
        if config['loss'] == 'mse':
          y = y.float()
        elif config['loss'] == 'bce':
          y = y.long()-1
        loss = criterion(pt, y)
        if tepoch is not None:
          tepoch.set_description(f"{experiment} Train Epoch {epoch}/{max_epochs}, loss: {loss}")
        # backprop
        loss.backward()
        # update the optimizer
        optimizer.step()
        # log training loss
        total_train_loss.append(loss.item())
    
    if not config['ray_tune']:
      grad_to_tb(writer, model, epoch, prefix=config['experiment'])

    total_train_loss = np.array(total_train_loss).mean()
    writer.add_scalar(f'train/loss', total_train_loss, epoch)
    ys = np.hstack(ys)
    if config['loss'] == 'bce':
      pts = np.concatenate(pts).argmax(axis=1)
    elif config['loss'] == 'mse':
      pts = np.concatenate(pts).round()
    f1 = f1_score(ys, pts, average='macro')
    writer.add_scalar(f'train/f1', f1, epoch)
    acc = accuracy_score(ys, pts)
    writer.add_scalar(f'train/acc', acc, epoch)

    model.eval()

    report = {}
    if 'loocv' not in config or not config['loocv']:
      test_metrics = eval(loaders, model, criterion, config, split='test', epoch=epoch, current_es=current_es, writer=writer)
      report['t_f1'] = test_metrics['f1']
      report['t_acc'] = test_metrics['acc']

    if 'holdout' in config['splits']:
      hold_metrics = eval(loaders, model, criterion, config, split='holdout', epoch=epoch, current_es=current_es, writer=writer)
      report['h_f1'] = hold_metrics['f1']
      report['h_acc'] = hold_metrics['acc']

    val_metrics = eval(loaders, model, criterion, config, split='val', epoch=epoch, current_es=current_es, writer=writer)
    report['v_f1'] = val_metrics['f1']
    report['v_acc'] = val_metrics['acc']

    #if not config['ray_tune']:
    #  teststr = ''
    #  if 'loocv' not in config or not config['loocv']:
    #    tesstr = f", test: {report['t_f1']}"
    #  print(f"  F1-Score val: {report['v_f1']}{teststr}, held: {report['h_f1']}")

    config['running_validation_loss'].append(val_metrics['loss'])
    if len(config['running_validation_loss']) > 10:
      if np.array(config['running_validation_loss']).mean() > 10:
        raise Exception(f"Validation loss is too high, is the learning rate or dropout too high?\n  Last network output was: {pts}")
      config['running_validation_loss'] = config['running_validation_loss'][1:]


    if 'ray_tune' in config and config['ray_tune']:
      ray.tune.report(
        epoch=epoch,
        loss=float(val_metrics['loss']),
        gpu=current_gpu,
        **report
      )

    # early stopping
    if best_val_loss != None and val_metrics['loss'] > best_val_loss:
      current_es += 1
      if current_es >= early_stopping:
        if best_model is not None:
          torch.save(best_model.state_dict(), f"{config['experiment_folder']}/model-ffnet-epoch_{epoch}-valloss_{val_metrics['loss']}.pt")
        break
    else:
      current_es = 0
      best_model = copy.deepcopy(model)
      best_val_loss = val_metrics['loss']


class SimpleModels():
  def __init__(self, parent_parsers=[]):
    parser = argparse.ArgumentParser(parents=parent_parsers, add_help=False)
    parser.add_argument('--module', type=str, default='ffnet', choices=['ffnet', 'gru', 'transformer', 'tgnn', 'transformer_gnn', 'projecttimeff', 'projectfeaturesff'], help='which module to use')
    parser.add_argument('--sparsify-mlp-input', action='store_true', help='whether to use sparseify mlp in the ffnet')
    parser.add_argument('--features', type=str, default='')
    parser.add_argument('--exclude-features', type=str, default='')
    parser.add_argument('--standard-scaler', action='store_true', help='whether to standardize the features')

    parser.add_argument('--loocv', action='store_true', help='whether to use leave one out cross validation')
    parser.add_argument('--loocv-train-split', type=float, default=0.8, help='the percentage of the data to use for training')
    parser.add_argument('--loocv-holdout', type=str, default='', help='which participant to hold out (otherwise search over participant)')

    parser.add_argument('--human-annotation-split', action='store_true', help='use human annotation splits')

    parser.add_argument('--ffnet-sizes', type=str, default='')

    parser.add_argument('--feature-embed-size', type=int, default=64, help='size of the feature embedding')

    parser.add_argument('--filter-to-timesteps', type=int, default=0, help='how many timesteps to filter to. 0 means no filtering')

    parser.add_argument('--temporal-hidden-dim', type=int, default=64, help='size of the temporal hidden dimensions')
    parser.add_argument('--temporal-embed-size', type=int, default=64, help='size of the temporal embedding')

    parser.add_argument('--transformer-intermediate-node-features', type=int, required=False, default=0, help='how many intermediate node features to use for pre-encoding GNN node features')
    parser.add_argument('--transformer-intermediate-edge-features', type=int, required=False, default=0, help='how many intermediate edge features to use for pre-encoding GNN edge features')
    parser.add_argument('--transformer-with-pos-enc', action='store_true', help='whether to use positional encoding in the transformer')
    parser.add_argument('--transformer-num-heads', type=int, default=4, help='number of heads in the transformer')
    parser.add_argument('--transformer-num-layers', type=int, default=2, help='number of layers in the transformer')

    parser.add_argument('--gru-bidirectional', action='store_true', help='whether to use bidirectional gru')
    parser.add_argument('--gru-all-hidden-states', action='store_true', help='whether to use all hidden states from the gru to the ffnet')

    parser.add_argument('--gnn-sizes', type=str, default='')
    parser.add_argument('--gnn-intermediate-node-features', type=int, default=0)
    parser.add_argument('--gnn-intermediate-edge-features', type=int, default=0)
    parser.add_argument('--gru-num-layers', type=int, default=2, help='number of layers in the gru')

    parser.add_argument('--aggregate-gnn-features-over', type=str, default='node', help='which features to include in the gnn ouput to the ff classification head')

    parser.add_argument('--label', type=str, default='0', help='label to train on')
    parser.add_argument('--category-count', type=int, default=5, help='how many categories to use for the output')

    parser.add_argument('--scale-output', action='store_true', help='scale the output to the number of categories')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--loss', type=str, choices=['mse', 'bce'])
    parser.add_argument('--early-stopping', type=int, default=50, help='how many epochs for ealy stopping')

    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='which activation function to use on all except the last layer')
    parser.add_argument('--last-activation', type=str, default='relu', choices=['sigmoid', 'softmax', 'relu', 'linear'], help='which activation function to use on the last layer')

    parser.add_argument('--mmaped-data', action='store_true', help='use mmaped data')
    parser.add_argument('--with-edge-attr', action='store_true', help='use edge attributes in the non-graph models')

    parser.add_argument('--ray-tune', action='store_true', help='use ray tune')

    parser.add_argument('--gpus-per-trial', type=float, default=0.5, help='how many gpus per trial')
    parser.add_argument('--cpus-per-trial', type=int, default=4, help='how many cpus per trial')
    parser.add_argument('--ray-grace-period', type=int, default=50, help='how many epochs to wait before early stopping')
    parser.add_argument('--ray-reduction-factor', type=float, default=2, help='keep only 25% of trials after each reduction')
    parser.add_argument('--ray-num-samples', type=int, default=32, help='how many trials to run')

    parser.add_argument('--ray-grid-search', action='store_true', help='use ray tune grid search')
    parser.add_argument('--ray-grid-search-lr', type=str, default=None, help='ray tune grid search lr')
    parser.add_argument('--ray-grid-search-batch-size', type=str, default=None, help='ray tune grid search batch size')
    parser.add_argument('--ray-grid-search-dropout', type=str, default=None, help='ray tune grid search dropout')
    parser.add_argument('--ray-grid-search-features', type=str, default=None, help='ray tune grid search features')
    parser.add_argument('--ray-grid-search-time', type=str, default=None, help='ray tune grid timesteps and hz')

    parser.add_argument('--dataset', type=str, default='all_pytorch_implicit_feedback', choices=['simple_implicit_feedback', 'simple'], help='dataset')
    parser.add_argument('--data-dir', type=str, default='', help='data dir')

    parser.add_argument('--detect-anomaly', action='store_true', help='detect anomaly')

    os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'

    self.args, unknown = parser.parse_known_args()

    # one model per person when using loocv
    people_ids = []
    if self.args.loocv:
      for folder in os.listdir(self.args.data_dir):
        if os.path.isdir(os.path.join(self.args.data_dir, folder)):
          people_ids.append(folder.split('-')[-1])

    if self.args.mode == 'train':
      self.now = int(time.time())
      self.experiment_folder = output_path(self.args, raise_on_exists=False)
      self.run_name = configure_logging(self.args, self.now)
      self.tensorboard_path = output_path(self.args, 'tensorboard', self.run_name)

    if self.args.ray_tune:
      config = {
        "ray_tune": True,
        "gpus_per_trial": self.args.gpus_per_trial,
        "cpus_per_trial": self.args.cpus_per_trial,
        "ray_grace_period": self.args.ray_grace_period,
        "ray_reduction_factor": self.args.ray_reduction_factor,
        "ray_num_samples": self.args.ray_num_samples,
      }
      if self.args.ray_grid_search:
        if self.args.ray_grid_search_lr:
          lr_grid_cells = [float(x) for x in self.args.ray_grid_search_lr.split(",")]
        else:
          lr_start = 1e-4
          lr_end = 1e-1
          lr_steps = 5
          lr_step = (lr_end - lr_start)/lr_steps
          lr_grid_cells = [round(x, 5) for x in torch.arange(lr_start, lr_end, lr_step).tolist()]
        config["learning_rate"] = ray.tune.grid_search(lr_grid_cells)
        if self.args.ray_grid_search_batch_size:
          config["batch_size"] = ray.tune.grid_search([int(x) for x in self.args.ray_grid_search_batch_size.split(",")])
        else:
          config["batch_size"] = ray.tune.grid_search([64, 128, 256, 512, 1024])
        if self.args.ray_grid_search_dropout:
          config["dropout"] = ray.tune.grid_search([float(x) for x in self.args.ray_grid_search_dropout.split(",")])
        if self.args.ray_grid_search_features:
          from datasets.simple_dataset import ALL_FEATURES
          features = ALL_FEATURES
          if self.args.features:
            features = self.args.features.split(";")
          if self.args.with_edge_attr:
            features += ['edge_attr']
          if self.args.exclude_features:
            if self.args.features:
              raise ValueError("Cannot use --features and --exclude-features at the same time when using --ray-grid-search-features")
            features = [x for x in features if x not in self.args.exclude_features.split(";")]
          if self.args.ray_grid_search_features == 'all':
            config["features"] = ray.tune.grid_search(features)
          elif self.args.ray_grid_search_features == 'permute':
            f = []
            for i in range(len(features)):
              f += [list(x) for x in itertools.combinations(features, i+1)]
            config["features"] = ray.tune.grid_search(f)
        if self.args.ray_grid_search_time:
          config["search_time"] = ray.tune.grid_search(self.args.ray_grid_search_time.split(";"))
        # one model per held-out person
        if self.args.loocv:
          if self.args.loocv_holdout and self.args.loocv_holdout.strip() != '':
            # hold out a specific person
            config["subject_under_test"] = self.args.loocv_holdout
          else:
            # one model per person
            config["subject_under_test"] = ray.tune.grid_search(people_ids)
        if ',' in self.args.label:
          config['label'] = ray.tune.grid_search([int(x) for x in self.args.label.split(",")])
      else:
        config["learning_rate"] = ray.tune.loguniform(1e-4, 1e-1)
        config["batch_size"] = ray.tune.choice([64, 128, 256, 512, 1024])
    else:
      config = {
        "learning_rate": self.args.lr,
        "batch_size": self.args.batch_size,
        "tune": False,
        "dropout": self.args.dropout,
      }

    if self.args.loocv:
      config['loocv'] = True
      config['loocv_train_split'] = self.args.loocv_train_split
      if 'subject_under_test' not in config:
        config['subject_under_test'] = self.args.loocv_holdout
    
    if self.args.human_annotation_split:
      config['human_annotation_split'] = True
      config['splits'] = ['train', 'val', 'test']
  
    if not self.args.ray_grid_search_features:
      if ';' in self.args.features:
        raise ValueError("Cannot use multiple --features sets (with ';') when not using --ray-grid-search-features")
      config["features"] = self.args.features if self.args.features else None
      config["exclude_features"] = self.args.exclude_features if self.args.exclude_features else None

    config['standard_scaler'] = self.args.standard_scaler

    config['detect_anomaly'] = self.args.detect_anomaly
    config['running_validation_loss'] = []

    config['activation'] = self.args.activation
    config['last_activation'] = self.args.last_activation
    config['scale_output'] = self.args.scale_output
    config['category_count'] = self.args.category_count

    config['data_dir'] = self.args.data_dir
    if not config['data_dir'] and not config['search_time']:
      raise Exception('data_dir or search_time is required')

    config['mmaped_data'] = self.args.mmaped_data

    if self.args.ffnet_sizes:
      config['ffnet_sizes'] = [int(x) for x in self.args.ffnet_sizes.split(',')]
    else:
      raise Exception('ffnet_sizes is required')

    if self.args.gnn_sizes:
      config['gnn_sizes'] = [int(x) for x in self.args.gnn_sizes.split(',')]

    config['feature_embed_size'] = self.args.feature_embed_size

    config['temporal_hidden_dim'] = self.args.temporal_hidden_dim
    config['temporal_embed_size'] = self.args.temporal_embed_size

    config['gnn_intermediate_node_features'] = self.args.gnn_intermediate_node_features
    config['gnn_intermediate_edge_features'] = self.args.gnn_intermediate_edge_features
    config['gru_bidirectional'] = self.args.gru_bidirectional
    config['gru_all_hidden_states'] = self.args.gru_all_hidden_states
    config['gru_num_layers'] = self.args.gru_num_layers

    config['transformer_with_pos_enc'] = self.args.transformer_with_pos_enc
    config['transformer_intermediate_node_features'] = self.args.transformer_intermediate_node_features
    config['transformer_intermediate_edge_features'] = self.args.transformer_intermediate_edge_features
    config['transformer_num_heads'] = self.args.transformer_num_heads
    config['transformer_num_layers'] = self.args.transformer_num_layers

    if self.args.aggregate_gnn_features_over:
      config['aggregate_gnn_features_over'] = self.args.aggregate_gnn_features_over.split(',')

    config['verbose'] = self.args.verbose
    config['timesteps'] = self.args.timesteps
    config['filter_to_timesteps'] = self.args.filter_to_timesteps
    config['device'] = self.args.device
    config['max_epochs'] = self.args.max_epochs
    config['loss'] = self.args.loss

    config['module'] = self.args.module
    config['dataset'] = self.args.dataset
    # if we're not searching over labels
    if 'label' not in config:
      if ',' in self.args.label:
        raise Exception('cannot search over labels without ray_grid_search')
      config['label'] = int(self.args.label)

    config['experiment'] = self.args.experiment
    config['experiment_folder'] = self.experiment_folder
    config['tensorboard_path'] = self.tensorboard_path
    config['loader_num_workers'] = self.args.loader_num_workers

    if self.args.ray_grid_search:
      if config['loader_num_workers'] > 2:
        raise RuntimeError("Are you sure you want to usemore than 2 loader_num_workers when using ray_grid_search?")

    config['ray_tune'] = self.args.ray_tune

    config['early_stopping'] = self.args.early_stopping

    self.config = config

  def train(self):
    if self.args.ray_tune:
      results = None
      # check and make sure the config is pickleable
      #for k, v in self.config.items():
      #  print(k)
      #  copy.deepcopy(v)
      oneconfig = self.config.copy()
      # fail if the validation loss is too high
      scheduler = ray.tune.schedulers.FIFOScheduler()
      #scheduler = ASHAScheduler(
      #  metric="loss",
      #  mode="min",
      #  max_t=oneconfig['max_epochs'],
      #  grace_period=oneconfig['ray_grace_period'],
      #  reduction_factor=oneconfig['ray_reduction_factor'])
      metric_columns = [
        "epoch",
        "loss",
        "gpu",
        "v_f1",
        "v_acc",
        "h_f1",
        "h_acc",
      ]
      if 'loocv' not in oneconfig or not oneconfig['loocv']:
        metric_columns += [
          "t_f1",
          "t_acc",
        ]
      reporter = CLIReporter(
        metric_columns=metric_columns
      )
      ray.init(
        include_dashboard=True,
        dashboard_port=8267,
        dashboard_host='0.0.0.0',
      )
      try:
        output_dir = f"/data/ray/{os.path.basename(oneconfig['experiment_folder'].strip('/'))}"
        print(f"output_dir: {output_dir}")
        tuner = tune.Tuner(
          tune.with_resources(
            tune.with_parameters(train_loop),
            resources={"cpu": oneconfig['cpus_per_trial'], "gpu": oneconfig['gpus_per_trial']},
          ),
          tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=oneconfig['ray_num_samples'],
            reuse_actors=True,
          ),
          run_config=ray.air.RunConfig(
            local_dir=output_dir,
            log_to_file=True,
            progress_reporter=reporter,
            failure_config=ray.air.FailureConfig(
              max_failures=3,
              fail_fast=False,
            )
          ),
          param_space=oneconfig,
        )
        results = tuner.fit()
      except Exception as e:
        print(e)
        traceback.print_exc(limit=None, file=None, chain=True)

      if results is not None:
        print(results)
        #best_trial = results.get_best_result("loss", "min", "last")
        #print("Best trial config: {}".format(best_trial.config))
        #print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        #print("Best trial final test macro F1 score: {}".format(best_trial.last_result["v_f1"]))

    else:
      train_loop(self.config)
