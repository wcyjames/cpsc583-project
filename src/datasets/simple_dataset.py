import os
from tqdm import tqdm
from os import path
from os.path import isfile, isdir, join
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import random

from torch_geometric.data import Data as GeometricData

from datasets.human_annotations import HUMAN_ANNOTATIONS

participant_list = [
    "R9pToyeea8", "a5uL6IEOeC", "PIlPrmr4j2", "5FzupRqB25", "21EwOyXqpl",
    "FA39VRkcda", "rF9yOs7nBl", "clE67wOvUh", "cJDeaswU83", "akQ0287c2M",
    "7Cd7eJdMG5", "WeTxzD8JpU", "UJf4seofEH", "7xKczr6tbD", "K3voJb36wH",
    "MieEKaWCKL", "fCNp4PedPe", "wv4sDUvijG", "VGEvcbeZe1", "lu1xnHdyWk",
    "kRbvOBFYA7", "I9R97KxWLE", "2Tuviyzdxc", "q5oM5GZs4t", "yAEzMrqG17",
    "7eZsbGRfa4", "wngtl0tMbP", "KEDVqr3Jfl", "5mDkOaLjJA", "LHXlLsFDdh",
    "btooytSyiE", "5aRxuFHgfz", "zBv92SRSbL", "xNXRmJHAY9", "d7XdKgF8AP",
    "yP42DLasnj", "YVOW7HypyL", "olnC55KTqL", "ukwIRZwSAk", "Zs7ycE4IaO",
    "i0PKwf5Grp", "UteqXm8J9d", "brcLiaO7VK", "Dt1pNDo7MC", "QKmk24MPEh",
    "QlcnGJFggP", "BTqpMH8MlI", "fVDZososVm", "cMcx9Ha611", "UPyCMeEKur",
    "4yyVIuJZ3r", "dmKfdvXaS8", "p2XiQOj75q", "y76nak9vm1", "u4OdfW5R8b",
    "Rx2Mwm8Cnw", "EpIuBp87fb", "bKhQ731DqL", "u6nho6pmEy", "LByV32kk1C"
]
# subset of features releated to implicit feedback
FEATURES = dict(
    # in prolific, the map includes physical features, spatial and gaze
    physical = ['map_resnet18'],
    spatial = ['goal', 'follower', 'nearby', 'gaze'],
    # in prolific:
    #  - visualization of the head feature has no roll and pitch
    #  - head -> only yaw, which is included in follower
    #facial=['eye', 'lip', 'head', 'gaze'],
    facial=['eye', 'lip'],
    viz = ['map', 'gaze_direction_local', 'gaze_direction_global', 'gaze_origin_local', 'gaze_origin_global', 'eye_data_validata_bit_mask_left', 'eye_data_validata_bit_mask_right', 'eye_data_validata_bit_mask_combined', 'convergence_distance_validity', 'convergence_distance_mm', 'tracking_improvements', 'tracking_improvements_length'],
)
# flatten the features
ALL_FEATURES = []
for k, feature in FEATURES.items():
    if k == 'viz':
        continue
    ALL_FEATURES += feature
# for O(1) lookup of features by child name
FEATURE_PARENTS = dict(
    map_resnet18 = 'physical',
    map = 'viz',
    goal = 'spatial',
    follower = 'spatial',
    nearby = 'spatial',
    eye = 'facial',
    lip = 'facial',
    head = 'facial',
    gaze = 'spatial',
    gaze_direction_local = 'viz',
    gaze_direction_global = 'viz',
    gaze_origin_local = 'viz',
    gaze_origin_global = 'viz',
    eye_data_validata_bit_mask_left = 'viz',
    eye_data_validata_bit_mask_right = 'viz', 
    eye_data_validata_bit_mask_combined = 'viz',
    convergence_distance_validity = 'viz',
    convergence_distance_mm = 'viz',
    tracking_improvements = 'viz',
    tracking_improvements_length = 'viz',
    behavior = 'other',
    time_elapsed = 'other',
    time_remaining = 'other',
)

class StandardScaler:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
    
    def transform(self, x):
        return (x - self.mean) / self.stddev
    
    def inverse_transform(self, x):
        return x * self.stddev + self.mean
    

class SimpleDataset(Dataset):

    def __init__(self, config=None, split='train', standard_scaler=None):
        """
        Args:
            subject_under_test: the subject will be evaluated on, use the
                                rest of other subjects under the same condition for train + test. 
            data_dir:           root directory saving all data files  
        """

        # must be before the call to featureizeone
        self.flattenedshapes = None
        self._all_splits = None

        if config is None:
            raise ValueError("config must be specified")
        self.config = config
        self.split = split
        data_dir = config['data_dir']
        if data_dir is None or not len(data_dir):
            raise ValueError("data_dir must be specified")
        if not os.path.exists(data_dir):
            raise ValueError(f"data_dir {data_dir} does not exist")
        print(f"Load data from {data_dir}")

        if 'loocv_train_split' in config:
            self.loocv_train_split = config['loocv_train_split']
        else:
            self.loocv_train_split = 0.8

        self.pt_files = self.find_pt_files()
        self.is_mmaped = False
        if 'mmaped_data' in config:
            self.is_mmaped = config['mmaped_data']

        # default to all node features
        self.features = ALL_FEATURES
        # setup features, if specified
        if 'features' in config and config['features'] is not None:
            self.features = config['features']
            if type(self.features) == list:
                features = []
                for feature in self.features:
                    features += feature.split(',')
                self.features = list(set(features))
            if type(self.features) == str:
                if not len(self.features.strip()):
                    raise ValueError("features must be a non-empty string")
                self.features = self.features.strip().split(',')
            # don't include edge attrs if we didn't specify it
            if 'edge_attr' not in self.features:
                self.with_edge_attr = False
        if 'exclude_features' in config and config['exclude_features'] is not None and type(config['exclude_features']) == str:
            excluded_features = config['exclude_features'].strip().split(',')
            for excluded_feature in excluded_features:
                if excluded_feature not in self.features:
                    raise ValueError(f"excluded feature {excluded_feature} not in features")
            self.features = [f for f in self.features if f not in excluded_features]
        # should we concatenate edge features for the flattened data?
        self.with_edge_attr = False
        if 'with_edge_attr' in config:
            self.with_edge_attr = config['with_edge_attr']

        print(f"Using features: {', '.join(self.features)} -- with_edge_attr: {self.with_edge_attr}")
        
        # featurizeone checks for this, need to set to none first 
        self.standard_scaler = None

        # load an example for the shapes
        self.shapes = None
        print(f"split: {split} -- len: {len(self.pt_files)}")
        self.featurizeone(self.pt_files[0])

        # setup standard scaler, must be done after featurizeone and memload
        if standard_scaler == 'compute':
            print("Computing standard scaler params")
            samples = []
            for pt in tqdm(self.pt_files):
                #TODO 
                X, _ = self.featurizeone(pt)
                samples.append(X)
            if type(samples[0]) == dict:
                samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
                if 'edge_index' in samples:
                    del samples['edge_index']
                self.standard_scaler = {k: StandardScaler(samples[k].mean(), samples[k].std()) for k in samples.keys()}
            else:
                samples = torch.cat(samples)
                self.standard_scaler = StandardScaler(samples.mean(), samples.std())
        else:
            self.standard_scaler = standard_scaler

        # load the dataset into mem, if requested
        self.mmap_data = None
        if self.is_mmaped:
            self.memload()
    
    def all_splits(self):
        if self._all_splits is None:
            self._all_splits = []
            for folder in os.listdir(self.config['data_dir']):
                if os.path.isdir(os.path.join(self.config['data_dir'], folder)):
                    self._all_splits.append(folder.split('-')[-1])
        return self._all_splits
    
    def find_pt_files(self):
        pt_files = None
        if 'loocv' in self.config and self.config['loocv']:
            subject_under_test = self.config['subject_under_test']
            if self.split == 'holdout':
                pt_files = sorted(list(glob.iglob(os.path.join(self.config['data_dir'], "*-"+ subject_under_test, '*.pt'))))
            else:
                all_pts = []
                for split in self.all_splits():
                    if split == subject_under_test:
                        print(f"holding out '{split}'")
                        continue
                    all_pts += sorted(list(glob.iglob(os.path.join(self.config['data_dir'], "*-"+ split, '*.pt'))))
                # deterministic shuffle, to make sure our split is the same each time this is called
                random.Random(42).shuffle(all_pts)
                #print(f"all_pts ({len(all_pts)}): {all_pts}")
                if self.split == 'train':
                    pt_files = all_pts[:int(len(all_pts)*self.loocv_train_split)]
                    print(f"train ({len(pt_files)})")
                elif self.split == 'val':
                    pt_files = all_pts[int(len(all_pts)*self.loocv_train_split):]
                    print(f"val ({len(pt_files)})")
                else:
                    raise ValueError(f"invalid split: {self.split}")
        
        elif 'human_annotation_split' in self.config and self.config['human_annotation_split']:
            if self.split not in ['train', 'val', 'test']:
                raise ValueError(f"invalid split: {self.split}")
            pt_files = []
            for participant_id, video_ids in HUMAN_ANNOTATIONS.items():
                for pt_file in list(glob.iglob(os.path.join(self.config['data_dir'], "*-"+participant_id, '*.pt'))):
                    video_id = int(pt_file.split('/')[-1].split('.pt')[0].split('_')[-1])
                    if self.split == 'test':
                        if video_id in video_ids:
                            pt_files.append(pt_file)
                        else:
                            # skip this sample, it's in the test set
                            pass
                    else:
                        if video_id in video_ids:
                            # don't include the sample, it's in the test set
                            pass
                        else:
                            pt_files.append(pt_file)

            rng = np.random.default_rng(seed=42)
            rng.shuffle(pt_files)

            if self.split == 'val':
                pt_files = pt_files[:int(len(pt_files)*0.2)]
            elif self.split == 'train':
                pt_files = pt_files[int(len(pt_files)*0.2):]
            
            print(f"human_annotation {self.split}: {len(pt_files)}")

        else:
            pt_files = list(glob.iglob(os.path.join(self.config['data_dir'], "*-"+self.split, '*.pt')))
        
        return pt_files


    def __len__(self):
        return len(self.pt_files)
    

    def memload(self):
        self.mmap_data = []
        with tqdm(self.pt_files) as tepoch:
            for i, pt in enumerate(tepoch):
                tepoch.set_description(f"Preprocessing {pt}")
                self.mmap_data.append(self.featurizeone(pt))

    def unflatten(self, batch):
        if self.with_edge_attr:
            raise NotImplementedError("unfeaturization for ffnet with edge_attr not implemented")
        if self.config['module'] == 'ffnet':
            unflattened = {}
            lastshape = np.array(0)
            for i, feature in enumerate(self.features):
                shape = np.array(self.flattenedshapes[i])
                unflattened[feature] = batch[:, lastshape.prod():lastshape.prod()+shape.prod()].reshape((-1, *shape))
                lastshape = shape
            return unflattened
        else:
            raise NotImplementedError(f"unfeaturization for module {self.config['module']} not implemented")

    def featurizeone(self, one_pt):
        idx = os.path.basename(one_pt).split('.')[0].split('_')[-1]
        one = torch.load(one_pt)
        # label
        y = one['y']
        if 'label' in self.config and self.config['label'] is not None:
            y = y[self.config['label']]

        graph = one['graph']
        num_timesteps = graph.x.shape[0]

        # setup dynamic feature sizes
        shapes = None
        if self.shapes is None:
            shapes = dict(
                num_timesteps = num_timesteps,
                node_features = graph.edge_attr.shape[1],
                edge_features = graph.edge_attr.shape[1],
            )
        
        # For analysis
        if self.config['module'] in ['analysis', 'projectfeaturesff']:
            X = {
                'idx': idx,
                'pt_file_name': one_pt
            }
            if shapes is not None:
                shapes['by_feature'] = {}
            for feature in self.features:
                X[feature] = one[FEATURE_PARENTS[feature]][feature]
                if type(X[feature]) == np.ndarray:
                    X[feature] = torch.tensor(X[feature])
                if (type(X[feature]) != torch.Tensor):
                    X[feature] = torch.tensor(X[feature])
                X[feature] = X[feature].float()
                if shapes is not None:
                    shapes['by_feature'][feature] = X[feature].shape
            if 'after_behavior_change' in one:
                X['after_behavior_change'] = int(one['after_behavior_change'])
            if shapes is not None:
                self.shapes = shapes
            return X, y

        # Simple MLP model
        # output shape: (num_features) -> flatten everything
        if self.config['module'] == 'ffnet':
            self.flattenedshapes = []
            features = []
            for feature in self.features:
                try:
                    one_feature = one[FEATURE_PARENTS[feature]][feature]
                    if feature == 'map':
                       one_feature = np.array(one) 
                    flat = one_feature.flatten()
                except AttributeError as ae:
                    print(ae)
                    print(f"Error with feature {feature} in {one_pt}")

                if type(flat) == np.ndarray:
                    flat = torch.tensor(flat)
                features.append(flat)
                self.flattenedshapes.append(flat.shape)
            X = torch.cat(features).flatten()
            if self.with_edge_attr:
                X = torch.cat((X, graph.edge_attr.flatten()))
            if shapes is not None:
                shapes['num_features'] = X.shape[0]
                self.shapes = shapes
            if self.standard_scaler is not None:
                X = self.standard_scaler.transform(X)
            X = X.float()
            if 'with_behavior_change' in self.config and self.config['with_behavior_change']:
                return X, y, int(one['after_behavior_change'])
            return X, y

        # Transformer model
        # time dimension first, all the rest of the features are concatenated
        # output shape: (num_timesteps, all_features) -> flatten all except time
        if self.config['module'] in ['transformer', 'gru', 'projecttimeff']:
            self.flattenedshapes = []
            features = []
            # concat along time dim
            for feature in self.features:
                flat = one[FEATURE_PARENTS[feature]][feature]
                if type(flat) == np.ndarray:
                    flat = torch.tensor(flat)
                flat = flat.flatten(1)
                features.append(flat)
                self.flattenedshapes.append(flat.shape)
            X = torch.cat(features, dim=1).float()

            if shapes is not None:
                shapes['num_features'] = X.shape[1]
                self.shapes = shapes
            # optionally add the edge attributes
            if self.with_edge_attr:
                X = torch.cat((X, graph.edge_attr.transpose(0, 2).flatten(1)), dim=1)
            if self.standard_scaler is not None:
                X = self.standard_scaler.transform(X)
            return X, y

        # GNN model
        if self.config['module'] == 'tgnn':
            if 'nearby' not in self.features:
                raise ValueError("'nearby' people feature must be included for GNN")

            # everything that is not "nearby" is going into the gga
            GGA_FEATURES = list(set(self.features) - set(['nearby']))
            
            gga_features = []

            # concat along time dim
            for feature in self.features:
                if feature == 'nearby':
                    continue
                flat = one[FEATURE_PARENTS[feature]][feature]
                if type(flat) == np.ndarray:
                    flat = torch.tensor(flat)
                flat = flat.flatten(1)
                # don't include global features here
                if feature in GGA_FEATURES:
                    gga_features.append(flat)
                else:
                    raise RuntimeError(f"Only 'nearby' can be a node feature, currently, so specify: '{feature}' as a GGA feature")
            if len(gga_features):
                gga = torch.cat(gga_features, dim=1).float()
                gga_shape = gga.shape
            else:
                gga = None
                gga_shape = None

            # by default graph.x is of shape (timesteps, num_nodes, num_features)
            # permute to (num_nodes,timesteps,num_features) and flatten, keeping only the node dimension
            X = graph.x.permute(1, 0, 2).flatten(1)

            # compute the node features in the shape (num_nodes, num_timesteps * num_features)

            # edge attrs if of shape (num_edges, num_timesteps, num_features)
            # flatten keeping only the number of edges
            edge_attr = graph.edge_attr.flatten(1)

            if shapes is not None:
                # this is the 'nearby' feature only
                shapes['num_nodes'] = X.shape[0]
                shapes['node_features'] = X.shape[1]
                # this is the 'nearby' feature, represented as the distance between nodes
                shapes['edge_features'] = edge_attr.shape[1]
                shapes['gga_features'] = gga_shape
                self.shapes = shapes
            #print(f"Shapes: {shapes}")
            #import pdb; pdb.set_trace(); 1

            # in qiping's gnn:
            #   implicit features are repeated per node
            #   no one_time features are used
            d = {
                # x shape is (num_nodes, num_features * num_timesteps)
                'x': X.float(),
                'edge_attr': edge_attr.float(),
                'edge_index': graph.edge_indices,
                'gga': gga.float(),
            }

            if self.standard_scaler is not None:
                for k in d:
                    if k == 'edge_index':
                        continue
                    d[k] = self.standard_scaler[k].transform(d[k])
            return d, y
        
        # Temporal GNN model
        if self.config['module'] == 'temporal_gnn':
            raise NotImplementedError("temporal GNN featurization not fully implemented")
            gga_features = []

            # concat along time dim
            for feature in self.features['facial']:
                
                flat = one[FEATURE_PARENTS[feature]][feature]
                if type(flat) == np.ndarray:
                    flat = torch.tensor(flat)
                flat = flat.flatten(1)
                # don't include global features here
                if feature in GGA_FEATURES:
                    gga_features.append(flat)
                else:
                    raise RuntimeError(f"Only 'nearby' can be a node feature, currently, so specify: '{feature}' as a GGA feature")
            if len(gga_features):
                gga = torch.cat(gga_features, dim=1).float()
                gga_shape = gga.shape
            else:
                gga = None
                gga_shape = None

            # by default graph.x is of shape (timesteps, num_nodes, num_features)
            X = graph.x

            # compute the node features in the shape (num_nodes, num_timesteps * num_features)

            # edge attrs if of shape (num_edges, num_timesteps, num_features)
            edge_attr = graph.edge_attr

            if shapes is not None:
                # this is the 'nearby' feature only
                shapes['num_nodes'] = X.shape[0]
                shapes['node_features'] = X.shape[1]
                # this is the 'nearby' feature, represented as the distance between nodes
                shapes['edge_features'] = edge_attr.shape[1]
                shapes['gga_features'] = gga_shape
                self.shapes = shapes
            #print(f"Shapes: {shapes}")
            #import pdb; pdb.set_trace(); 1

            d = {
                # x shape is (num_nodes, num_features * num_timesteps)
                'x': X.float(),
                'edge_attr': edge_attr.float(),
                'edge_index': graph.edge_indices,
                'gga': gga.float(),
            }

            if self.standard_scaler is not None:
                for k in d:
                    if k == 'edge_index':
                        continue
                    d[k] = self.standard_scaler[k].transform(d[k])
            return d, y

        # Transformer then GNN
        if self.config['module'] == 'transformer_gnn':
            raise NotImplementedError("Transformer then GNN featurization not implemented")
            if shapes is not None:
                shapes['num_features'] = X.shape[1]
                shapes['num_nodes'] = X.shape[0]
                shapes['node_features'] = X.shape[1]
                shapes['edge_features'] = graph.edge_attr.shape[1]
                self.shapes = shapes

            if self.standard_scaler is not None:
                X = self.standard_scaler.transform(X)
            # in qiping's gnn:
            #   implicit features are repeated per node
            #   no one_time features are used
            return {
                # x shape is (num_nodes, num_features * num_timesteps)
                'x': X,
                'edge_index': graph.edge_index,
                'edge_attr': graph.edge_attr,
            }, y

        raise RuntimeError(f"Unknown module type: '{self.config['module']}'")

    def __getitem__(self, idx):
        # use the mmap file
        if self.mmap_data is not None:
            #print(f"loading {idx} from mmap, xs: {self.mmap_data['xs'].shape}, ys: {self.mmap_data['ys'].shape}")
            return self.mmap_data[idx]
        
        # otherwise load and preprocess now
        return self.featurizeone(self.pt_files[idx])
  
if __name__ == '__main__':
    # pre-cache the mmap file
    for split in ['train', 'val', 'test', 'holdout']:
        SimpleDataset(config={'is_mmaped': True}, split=split)
        print(f"Loaded {split}")


    