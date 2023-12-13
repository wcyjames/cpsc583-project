import os
import pathlib
import logging
import glob

import json
import pandas as pd

CLASS_NAMES = [
    'cross_path',
    'down_path',
    'empty',
    'join_group',
    'leave_group'
]
MAP_NAMES = {
  # lab scenes
  'lab': 'lab',
  'labstudy': 'labstudy',
  'agentcontrollab': 'lab',
  'labscenegraph' : 'lab',
  # warehouse scenes
  'smallwarehouse': 'warehouse_small',
  'agentcontrolsmallwarehouse': 'warehouse_small',
  'smallwarehousescenegraph':'warehouse_small',
  # outdoor scenes
  'outdoor': 'outdoor',
  'agentcontroloutdoor': 'outdoor',
  'outdoorscenegraph' : 'outdoor',
  'hotel' : 'hotel',
  'university' : 'university',
  'eth' : 'ETH',
  'zara' : 'zara',
}

def output_path(args, *pth, fname='', raise_on_exists=False):
    ret = '/'.join([args.output, 'running', args.experiment]+list(pth))
    experiment_folder = pathlib.Path(ret)
    if raise_on_exists and pathlib.Path(experiment_folder).exists():
        raise Exception(f"\n\nOutput path {experiment_folder} exists\n  rm -r {experiment_folder}\nand re-run to ignore\n\n")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    return os.path.join(ret, fname)

def configure_logging(args, now, train_or_run='learner'):
    pathlib.Path(output_path(args)).mkdir(parents=True, exist_ok=True)
    if hasattr(args, 'loss'):
        run_name = f"{args.loss}-lr_{args.lr}_{now}"
    else:
        run_name = f"{now}"
    log_path = output_path(args, fname=f"{train_or_run}-{run_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fm = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fm)
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info(f"Configured logging to output to: {log_path} and terminal")
    return run_name

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_bags(args):
  source_paths = args.source_paths.split(',')
  print(f"bag path: {source_paths}")
  folders = []
  for bag_path in source_paths:
    if os.path.isdir(bag_path):
      folders.append(bag_path)
    for o in os.listdir(bag_path):
      full_o = os.path.join(bag_path, o)
      if os.path.isdir(full_o):
        folders.append(full_o)
  # read bags
  globs = []
  bags = []
  for folder in folders:
    _glob = os.path.join(folder, '*.bag')
    _bags = list(glob.iglob(_glob))
    print(f"looking in: {_glob}, found {len(_bags)} bags")
    if len(list(_bags)) == 0:
      _glob = os.path.join(folder, '*/*.bag')
      _bags = list(glob.iglob(_glob))
      print(f"looking in: {_glob}, found {len(_bags)} bags")
    if len(list(_bags)) == 0:
      raise RuntimeError(f"No bags found in {folder}")
    globs.append(_glob)
    bags += _bags

  baglist = list(set(bags))
  baglist.sort()
  length = len(baglist)
  print(f"========== {length} bags found in {', '.join(globs)} ==========")
  return baglist

def find_best_model(experiment_folder):
  best_loss_val = None
  best_model = None
  modelpaths = list(glob.iglob(os.path.join(experiment_folder, 'model-*.pt')))
  if len(modelpaths) == 0:
    raise RuntimeError(f"Could not find any models in {experiment_folder}")
  for modelp in modelpaths:
    tipe, epoch, vl = modelp.split('/')[-1].split('.pt')[0].split('model-')[-1].split('-')
    loss_val = float(vl.split('_')[-1])
    epoch = int(epoch.split('_')[-1])
    if loss_val is None:
      raise RuntimeError(f"Could not find valloss in {modelp} model name")
    if best_loss_val is None or best_loss_val > loss_val:
      best_loss_val = loss_val
      best_model = modelp
      best_epoch = epoch
  print(f"Found best model {best_model} with val loss {best_loss_val} from epoch {best_epoch}")
  return best_model

def load_ray(experiments: dict, best_by='loss'):
  stats = {
    'loaded': [],
    'empty': [],
  }
  results = []
  for experiment, details in experiments.items():
    json_files = list(glob.iglob(f"/data/ray/{experiment}/*/*/result.json"))
    print(f"found {len(json_files)} files for experiment '{experiment}'")
    found_pts = 0
    for jsonf in json_files:
      globpath = os.path.join(os.path.dirname(jsonf), "*/*/*/*.pt")
      pt = list(glob.iglob(globpath))
      if len(pt) == 0:
        print(f"no models found for '{jsonf}' in '{globpath}'")
        pt = None
      else:
        pt = pt[0]
        found_pts += 1
      one_experiment = []
      with open(jsonf, 'r') as f:
        for line in f.readlines():
          one_epoch = json.loads(line)
          one_epoch['experiment'] = experiment
          if 'note' in details:
            one_epoch['note'] = details['note']
            one_epoch['type_and_note'] = f"{details['type']}_{details['note']}"
          else:
            one_epoch['note'] = ''
            one_epoch['type_and_note'] = details['type']
          config = one_epoch['config']
          del one_epoch['config']
          for k,v in config.items():
            one_epoch[f"config_{k}"] = v
          for k,v in details.items():
            one_epoch[f"experiment_{k}"] = v
          one_epoch['pt'] = pt
          one_experiment.append(one_epoch)
      #print(f"found {len(pt)} models for experiment '{experiment}'")
      _df = pd.DataFrame(one_experiment)
      if best_by not in _df:
        stats['empty'].append(jsonf)
        continue
      stats['loaded'].append(jsonf)
      if best_by == 'loss':
        best_epoch = _df.iloc[_df[best_by].idxmin()]
      else:
        best_epoch = _df.iloc[_df[best_by].idxmax()]
      results.append(best_epoch)
  print(f"loaded {found_pts} pt files for {len(stats['loaded'])} experiments, {len(stats['empty'])} empty")
  return pd.DataFrame(results), stats