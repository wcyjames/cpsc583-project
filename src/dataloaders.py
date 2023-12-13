import os
import glob
import json
import torch

def data_split_loaders(dataset_type, sample_list_to_graph_function, args, splits=dict(train=0.8, test=0.1, val=0.1), seed=42):
  from torch_geometric.loader import DataLoader
  dataparams = {}
  if 'shuffle' not in dataparams:
    dataparams['shuffle'] = True
  if 'batch_size' not in dataparams:
    dataparams['batch_size'] = args.batch_size
  if 'num_workers' not in dataparams:
    dataparams['num_workers'] = args.loader_num_workers
  # per: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
  if 'pin_memory' not in dataparams:
    dataparams['pin_memory'] = True

  # load all metadata first
  count = 0
  samples = []
  for path in args.data_paths.split(','):
    with open(os.path.join(path, 'metadata.json'), 'r') as f:
      metadata = json.load(f)
    count += metadata['count']
    samples += metadata['samples']

  counts = {
    'train': int(round(splits['train']*count)),
    'test': int(round(splits['test']*count)),
    'val': int(round(splits['val']*count))
  }

  # make counts match by rounding test up or down if necessary
  agg_count = sum(counts.values())
  diff = count - agg_count
  counts['test'] += diff

  # split metadata['samples'] into train/test/val
  # use a fixed random seed
  # TODO: replace with a context manager to make sure this doesn't have downstream effects!
  # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  torch.manual_seed(seed)
  train, test, val = torch.utils.data.random_split(samples, [
      counts['train'],
      counts['test'],
      counts['val']
  ])
  # reset the random seed
  torch.manual_seed(torch.initial_seed())
  # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

  train_metadata = {'count': counts['train'], 'samples': train}
  test_metadata = {'count': counts['test'], 'samples': test}
  val_metadata = {'count': counts['val'], 'samples': val}

  train = DataLoader(dataset_type(sample_list_to_graph_function, args, train_metadata, 'train'), **dataparams)
  test = DataLoader(dataset_type(sample_list_to_graph_function, args, test_metadata, 'test'), **dataparams)
  val = DataLoader(dataset_type(sample_list_to_graph_function, args, val_metadata, 'val'), **dataparams)

  return {
    'train': train,
    'test': test,
    'val': val
  }

def implicit_pytorch_split_loaders(config):
  from torch.utils.data import DataLoader
  from torch_geometric.loader import DataLoader as PyGDataLoader
  from datasets.implicit_feedback_pytorch_dataset import ImplicitFeedbackPytorchDataset
  from datasets.simple_dataset import SimpleDataset
  from datasets.mpgnn_dataset import MPGNNDataset
  batch_size = config['batch_size']
  loader_num_workers = config['loader_num_workers']
  dataset = config['dataset']

  # allow search over the time configurations
  if 'search_time' in config and config['search_time']:
    timesteps, hz = config['search_time'].split(',')
    config['data_dir'] = f"/data/test_gnn/sean-vr/datasets/sean-vr-nearby-720_cm-{timesteps}_ts-hz_{hz}-16_agents"
    config['timesteps'] = int(timesteps)

  dataparams = {}
  if 'shuffle' not in dataparams:
    dataparams['shuffle'] = True
  if 'batch_size' not in dataparams:
    dataparams['batch_size'] = batch_size
  if 'num_workers' not in dataparams:
    dataparams['num_workers'] = loader_num_workers
  # per: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
  if 'pin_memory' not in dataparams:
    dataparams['pin_memory'] = True

  if 'standard_scaler' in config and config['standard_scaler'] and dataset != 'simple':
    raise Exception(f"--standard-scaler only supported for simple, not {dataset}")
  
  loaders = {}

  if 'splits' in config and config['splits']:
    splits = config['splits']
    if type(splits) is str:
      splits = splits.split(',')
  else:
    splits = ['train', 'test', 'val', 'holdout']
    folders = []
    for f in glob.iglob(config['data_dir'] + '/*'):
      if not os.path.isdir(f):
        continue
      folders.append(os.path.basename(f).split('-')[-1])
    if len(folders) > len(splits):
      splits = folders
    config['splits'] = splits


  if dataset == 'all_pytorch_implicit_feedback':
    for split in splits:
      loaders[split] = DataLoader(ImplicitFeedbackPytorchDataset(split, config=config), **dataparams)
  elif dataset == 'simple':
    standard_scaler = None
    if 'loocv' in config and config['loocv']:
      for _split in ['train', 'val', 'holdout']:
        dataset = SimpleDataset(config=config, split=_split, standard_scaler=standard_scaler)
        loaders[_split] = DataLoader(dataset, **dataparams)
    else:
      # for non-loocv
      for split in splits:
        if 'standard_scaler' in config and config['standard_scaler'] and standard_scaler is None:
          if split != 'train':
            raise Exception('standard_scaler must be retrieved from train split first. make sure train is first in splits')
          standard_scaler = 'compute'
        dataset = SimpleDataset(config=config, split=split, standard_scaler=standard_scaler)
        if 'standard_scaler' in config and config['standard_scaler'] and split == 'train' and standard_scaler is None:
          standard_scaler = dataset.standard_scaler
        loaders[split] = DataLoader(dataset, **dataparams)
  elif dataset == 'qzs_gnn_graph_dataset':
    for split in splits:
      loaders[split] = PyGDataLoader(MPGNNDataset(config=config, split=split), **dataparams)
  else:
    raise Exception(f"dataset {dataset} not supported")

  return loaders

def configured_implicit_dataset_loaders(config):
  ''' An example dataset for analysis'''
  timesteps = 40
  hz = 5
  ds_date = '2023-08-23'
  if 'ds_date' in config and config['ds_date']:
    ds_date = config['ds_date']
  config = {**dict(
      batch_size = 1,
      loader_num_workers = 0,
      dataset = 'simple',
      data_dir = f"/data/test_gnn/sean-vr/datasets/by-participant-720_cm-16_agents-{timesteps}_ts-hz_{hz}-{ds_date}",
      timesteps = timesteps,
      with_edge_attr = False,
      module = 'analysis',
  ), **config}
  return implicit_pytorch_split_loaders(config)

