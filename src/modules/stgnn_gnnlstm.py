import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from modules.gnn.gnn import *
from utils import *

# static graph, no edge and global update
class GNN_LSTM(torch.nn.Module):
    """
    Similar to the Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, periods = 40, out_channels = 256):
        super(GNN_LSTM, self).__init__()

        self.periods = periods
        self.node_embedding_size = 10  # Set your desired embedding size

        self.node_features = 90
        self.num_nodes = 16
        self.gnn_out_size = 256 
        self.lstm_hidden_size = 128
        self.dropout = 0

        # single graph attentional layer with 4 attention heads
        # self.gat = GATConv(in_channels=self.node_features, out_channels=self.node_features,
        #     heads=4, dropout=0, concat=False)
        self.conv = GCNConv(in_channels=self.node_features, out_channels=self.gnn_out_size)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.num_nodes*self.gnn_out_size, hidden_size=self.lstm_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=self.lstm_hidden_size, hidden_size=out_channels, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        self.regression = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, items):
        # batch_size, time_steps, num_nodes, node_feature_dim
        x = items['x']
        edge_index = items['edge_index']
        # torch.Size([4, 40, 16, 90])
        batch_size, num_timesteps, num_nodes, num_features = x.shape 
        # x = x.squeeze(0).reshape(num_timesteps*num_nodes, num_features) #[640 = 16*40, 90]
        # edge_index = edge_index.squeeze(0)
        # x = self.gat(x, edge_index) # -> [640 = 16*40, 90]

        # import pdb;pdb.set_trace()

        # Approach 1: If use GCN, which supports static graph 
        # Ref: https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
        # [batch_size * num_nodes, num_features, num_timesteps]

        x = x.permute(0, 2, 3, 1).contiguous() # batch_size, num_nodes, time_steps, node_feature_dim
        x = x.reshape(batch_size * num_nodes, num_features, num_timesteps)
        edge_index = edge_index.reshape(2, -1) #[2,960]
        x = x.permute(0, 2, 1) # -> [batch_size * num_nodes, num_timesteps, num_features]
        x = self.conv(x, edge_index) # -> [batch_size * num_nodes, num_timesteps, gnn_output_size]

        # Approch 2: If use GAT, which does not support static graph
        # ref: https://github.com/pyg-team/pytorch_geometric/issues/2844
        # RNN: 2 LSTM
        # [batchsize*n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
        

        # RNN: 2 LSTM
        # for lstm: x should be (seq_length, batch_size, n_nodes)
        x = x.permute(1, 0, 2).reshape(num_timesteps, batch_size, -1) 
        # [40, batch_size, 16*output_dim] -> [40, batch_size, lstm_hidden_dim]
        x, _ = self.lstm1(x)
        # [40, batch_size, lstm_hidden_dim] 
        # torch.Size([40, 4, 256])
        x, (h_n, c_n) = self.lstm2(x) #hn torch.Size([1, 4, 256])
        h = h_n[-1] # the last prediction

        h = self.regression(h)
        h = 4.0 * torch.sigmoid(h) + 1.0
        return h