#!/usr/bin/env python3

import argparse
import torch

from models import *

def main():
  parser = argparse.ArgumentParser(description='')

  ''' Mode / Model '''
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'rollout'])
  parser.add_argument('--model', type=str, required=True, choices=['foundation', 'controller', 'implicitfeedback', 'plaingnn',
                                                                   'simple', 'implicit_gnn', 'gru_gnn'])
  parser.add_argument('--foundation-model', type=str, default='')

  ''' Data '''
  parser.add_argument('--data-paths', type=str, help='comma separated path list of extracted SEAN data paths')
  parser.add_argument('--timesteps', type=int, default=5, help='how many past timesteps to process, must match preprocessing')
  parser.add_argument('--trajectory-timesteps', type=int, default=5, help='how many future timesteps to process, must match preprocessing')
  parser.add_argument('--use-follower', action='store_true', help='use follower data')
  

  ''' Output '''
  parser.add_argument('--output', type=str, default='experiments', help='path to output directory')

  ''' Occupancy grid Model '''
  parser.add_argument('--occ-input', action='store_true', help='use local occupancy grid map as an input')

  ''' Training params '''
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--loader-num-workers', type=int, default=12)
  parser.add_argument('--lr', type=float, default=1e-3, help='default learning rate')
  parser.add_argument('--max-epochs', type=int, default=200, help='maximum number of epochs')
  parser.add_argument('--with-attention', action='store_true', help='use attention-based model')
  parser.add_argument('--early-stopping-epochs', type=int, default=50, help='number of epochs after no improvement for early stopping')

  ''' Housekeeping '''
  parser.add_argument('--experiment', type=str, required=True, help='experiment name')

  ''' Hardware '''
  parser.add_argument('--device', type=str, default='cuda')

  ''' Debug '''
  parser.add_argument('--verbose', action='store_true', help='more verbose logging')
  parser.add_argument('--parallelism', type=int, default=24, help='parallelism of data processing')

  # parse
  args, unknown = parser.parse_known_args()

  if args.device.startswith("gpu:") and not torch.cuda.is_available():
    raise RuntimeError(f"CUDA not available for device {args.device}")

  model = None

  if args.model == 'foundation':
    from models.foundation_model import FoundationModel
    model = FoundationModel(parent_parsers=[parser])
  elif args.model == 'controller':
    from models.controller_model import ControllerModel
    model = ControllerModel(parent_parsers=[parser])
  elif args.model == 'implicitfeedback':
    from models.implicitfeedback_model import ImplicitFeedbackModel
    model = ImplicitFeedbackModel(parent_parsers=[parser])
  elif args.model == "plaingnn":
    from models.plain_gnn_model import PlainGNNModel
    model = PlainGNNModel(parent_parsers=[parser])
  elif args.model == 'simple':
    from models.implicit.simple_models import SimpleModels
    model = SimpleModels(parent_parsers=[parser])
  elif args.model == 'implicit_gnn':
    from models.implicit.gnn_model import ImplicitGNNModel
    model = ImplicitGNNModel(parent_parsers=[parser])
  elif args.model == 'gru_gnn':
    from models.gru_gnn_model import GRUGNNModel
    model = GRUGNNModel(parent_parsers=[parser])
  else:
    raise RuntimeError(f"Unknown loader {args.loader}")

  if args.mode == 'train': 
    model.train()
  elif args.mode == 'rollout': 
    from nodes.rollout import main
    main(args, model, parent_parsers=[parser])
  else:
    raise RuntimeError(f"Unknown mode {args.mode}")

if __name__ == "__main__":
  main()