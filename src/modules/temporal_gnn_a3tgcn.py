import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import *
from torch_geometric_temporal.nn.recurrent import A3TGCN2

# static graph, no edge and global update
class TemporalGNN_A3TGCN(torch.nn.Module):
    def __init__(self, batch_size=4, periods = 40, out_channels = 256):
        super(TemporalGNN_A3TGCN, self).__init__()

        self.batch_size = batch_size
        self.periods = periods
        self.node_embedding_size = 10  # Set your desired embedding size

        # Node feature encoder (MLP)
        # self.node_mlp = nn.Sequential(
        #     nn.Linear(82, 128),  # Set your desired output size
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 256)
        # )

        # Node binary feature encoder (Embedding)
        # self.node_embedding1 = nn.Embedding(2, self.node_embedding_size)
        # self.node_embedding2 = nn.Embedding(2, self.node_embedding_size)

        # self.node_features = 256 + 2 * self.node_embedding_size + 512
        self.node_features = 90

        self.tgnn = A3TGCN2(in_channels=self.node_features, 
                            out_channels=out_channels, 
                            periods=periods,
                            batch_size=batch_size)

        # config['osize'] = 5
        self.regression = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, items):

        # batch_size, time_steps, num_nodes, node_feature_dim -> batch_size, num_nodes, node_feature dim, time_steps
        node_sequence = items['x']
        edge_index = items['edge_index'][0]

        x = node_sequence.permute(0, 2, 3, 1).contiguous()
        # edge_index [2, n_edges per ]
        h = self.tgnn(x, edge_index)  # [b, n_node, n_feature, d_e] -> [b, n_node, d_hid]
        # import pdb;pdb.set_trace()
        
        h = h[torch.arange(h.size(0)), items['follower_ids'], :]
        # h = self.regression(h[:, -1]) # use the feature of the follower to predict
        h = self.regression(h)
        h = 4.0 * torch.sigmoid(h) + 1.0
        
        return h