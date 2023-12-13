import torch
from modules.ff import FFNet

from torch_geometric.data import Data
from torch_geometric.data import Batch

from torch_geometric.nn import MessagePassing

class MessagePassingLayer(MessagePassing):
    '''
    See for detail: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html

    Message Passing GNN is:

      $\mathbf{x}_{i}^{(k)}=\gamma^{(k)} (\mathbf{x} _{i}^{(k-1)}, \square _{j \in \mathcal{N}(i)} \phi^{(k)}(\mathbf{x} _{i}^{(k-1)}, \mathbf{x} _{j}^{(k-1)}, \mathbf{e} _{i, j}))$

    where:
      - $x^(k−1)$ is the node features of node $i$ in layer ($k$−1)
      - $e_{j,i} \in R^D$ are the edge features from node $j$ to node $i$
      - $\square$: aggregation method (permutation invariant function). i.e., mean, sum, max
      - $\gamma$, $\phi$: differentiable functions, such as MLP
    '''
    def __init__(self,
                 aggr='sum',
                 node_features=0, edge_features=0, internal_sizes=None,
                 phi_output_size=None, gamma_output_size=None, gga_features=None,
                 dropout=0.5):
        super().__init__(aggr=aggr)
        
        if internal_sizes is None: # single hidden layer of size node_features
            internal_sizes = [node_features]

        gga_features = 64
        # edge update input size is equal to the $x_j$, $x_i$, and $e_{j,i}$
        phi_input_size = node_features + node_features + edge_features + gga_features # 256

        # make the output of phi the same size as a node embedding
        if phi_output_size is None or phi_output_size == 0:
            phi_output_size = edge_features
        self.phi_output_size = phi_output_size
        # gamma's input is $x_i^{k-1}$ and $\bigoplus$
        gamma_input_size = node_features + phi_output_size + gga_features
        # output of gamma is the tuple (node_features, edge_features)
        if gamma_output_size is None or gamma_output_size == 0:
            gamma_output_size = node_features
        self.gamma_output_size = gamma_output_size

        # the edge update function
        self.phi = FFNet(
            isize=phi_input_size,
            osize=phi_output_size,
            internal_sizes=internal_sizes,
            dropout=dropout
        )
        # the node update function
        self.gamma = FFNet(
            isize=gamma_input_size,
            osize=gamma_output_size,
            internal_sizes=internal_sizes,
            dropout=dropout
        )

        # input of gamma_phi is the tuple (node_features, edge_features)
        # the global update function
        self.agg_edge = torch.nn.Linear(self.embed_dim, self.num_edges)
        self.agg_node = torch.nn.Linear(self.node_input_feature_dim, self.embed_dim)
        self.phi_global = FFNet(
            isize=gamma_output_size,
            osize=gga_features,
            internal_sizes=internal_sizes,
            dropout=dropout
        )
    
    def __repr__(self):
        res = f"phi: {self.phi}\n"
        res += f"gamma: {self.gamma}\n"
        return res

    def forward(self, x, edge_index, edge_attr, gga):
        '''
        batched inputs:
          - x: (num_nodes, num_node_features) node features
          - edge_index: (2, num_edges) edge index adjacency matrix
          - edge_attr: (num_edges, num_edge_features) edge attributes
        calls:
          - self.message: construct the message of node pairs x_i, x_j
          - self.aggregate: aggregate all messages of neighbors
          - self.update: update the embedding of node i with the aggregated message
        '''
        node_weights, edge_weights = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, gga=gga)

        # global update
        import pdb;pdb.set_trace()
        # node_weights torch.Size([5120, 64])
        # edge_weights torch.Size([76800, 64])
        # gga torch.Size([320, 64])
        aggregation_data = torch.cat((node_weights, edge_weights, gga), dim=1)
        updated_gga = self.phi_global(aggregation_data)

        return node_weights, edge_weights, updated_gga

    def message(self, x_i, x_j, edge_attr, x, gga):
        '''
        edge update function:
          combine $x_i^{k-1}$, $x_j^{k-1}$, $e_{j,i}$ via $\phi^{(k)}$
          batch dimension (ie. the first dimension) is $k$ where $k-1$ is the previous layer
            x_i: (num_edges, num_node_features)
            x_j: (num_edges, num_node_features)
            edge_attr: (num_edges, num_edge_features)
            x: (num_nodes, num_node_features)
            gga: (num_globl_features)
        return: phi which is aggregated into $\bigoplus$ by pytorch geometric
        '''
        # import pdb;pdb.set_trace()
        # gga torch.Size([320=batch_size*timesteps, 64])
        edge_gga = gga.unsqueeze(0).expand(240, gga.shape[0], gga.shape[1]).reshape(-1, gga.shape[-1])
        aggregation_data = torch.cat((x_i, x_j, edge_attr, edge_gga), dim=1) 
        # aggregation_data = torch.cat((x_i, x_j, edge_attr, gga), dim=1)
        phi = self.phi(aggregation_data) # -> torch.Size([76800 = batch_size*timesteps*num_edges, 64])
        
        return phi

    def aggregate(self, inputs, index, ptr, dim_size):
        agg = super().aggregate(inputs, index, ptr, dim_size)
        # return the aggregation and phi
        return agg, inputs

    def update(self, agg_out, x, gga):
        '''
        node update function:
            combine $x_i^(k-1)$, $\bigoplus$ (ie. agg_out) into the node representation for layer $k$
            agg_out is the output of the aggregation function
            combines the aggregated edge update with the node embedding and returns the updated node and edge embeddings
        '''
        agg, phi = agg_out
        # aggregation_data = torch.cat((x, agg), dim=1)
        node_gga = gga.unsqueeze(0).expand(16, gga.shape[0], gga.shape[1]).reshape(-1, gga.shape[-1])
        aggregation_data = torch.cat((x, agg, node_gga), dim=1)
        gamma = self.gamma(aggregation_data)
        # import pdb;pdb.set_trace()
        return gamma, phi


class STGNN_MPGNN_NODE_GLOBAL(torch.nn.Module):
    ''' mpgnn with optional gga that embeds features over time'''
    def __init__(self,
                 gga_dim=None,
                 batch_size=None,
                 temporal_embed_dim=None,
                 timesteps=None,
                 # number of input node features
                 node_features=None,
                 # number of node features between last gnn layer and mlp
                 intermediate_node_features=None,
                 edge_features=None,
                 # number of edge features between last gnn layer and mlp
                 intermediate_edge_features=None,
                 num_nodes=None, osize=1, gnn_internal_sizes=[64], ffnet_internal_sizes=[512, 256, 64], aggregate_over=[], dropout=0.2):
        super().__init__()

        self.gga_dim = gga_dim # torch.Size([40, 512])
        self.embed_dim = temporal_embed_dim
        self.embed_dim = 64

        self.node_input_feature_dim = 90
        self.edge_input_feature_dim = 4

        self.mp1 = MessagePassingLayer(
            node_features=self.embed_dim,
            edge_features=self.embed_dim,
            internal_sizes=gnn_internal_sizes,
            dropout=dropout
        )
        self.mp2 = MessagePassingLayer(
            node_features=self.embed_dim,
            edge_features=self.embed_dim,
            internal_sizes=gnn_internal_sizes,
            phi_output_size=self.embed_dim,
            gamma_output_size=intermediate_node_features,
            dropout=dropout
        ) 
        
        self.node_embedding = torch.nn.Linear(self.node_input_feature_dim, self.embed_dim)
        self.edge_embedding = torch.nn.Linear(self.edge_input_feature_dim, self.embed_dim)
        self.gga_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.gga_dim[-1], 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, self.embed_dim)
        )
        
        # Add LSTM
        self.num_nodes = 16
        self.num_edges = 240
        self.gnn_out_size = self.embed_dim
        self.lstm_out_size = self.embed_dim

        self.node_lstm = torch.nn.LSTM(input_size=self.gnn_out_size, hidden_size=self.lstm_out_size, num_layers=1)
        self.edge_lstm = torch.nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, num_layers=1)
        
        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_out_size, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 5)
        )


    def __repr__(self):
        res = f"mp1: {self.mp1}\n"
        res += f"mp2: {self.mp2}\n"
        return res

    def forward(self, items):
        # node embedding over time
        x = items['x']
        batch_size, timesteps, num_nodes, node_input_feature_dim = x.shape # torch.Size([4, 40, 16, 90])

        e = items['edge_attr']
        batch_size, num_edges, timesteps, edge_input_feature_dim = e.shape # torch.Size([4, 240, 40, 4])

        x_embed = self.node_embedding(x)
        edge_embed = self.edge_embedding(e)

        x = x_embed.reshape(batch_size*timesteps, num_nodes, self.embed_dim)
        e = edge_embed.reshape(batch_size*timesteps, num_edges, self.embed_dim)
        graphs = []
 
        for i in range(batch_size*timesteps):
            node_x = x[i] # torch.Size([16, 90])
            edge_index = items['edge_index'][0] # torch.Size([2, 240])
            edge_attr = e[i] # torch.Size([40, 4])
            graphs.append(Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(graphs)
        x = batch.x # (batch_size)*timesteps*num_nodes, node_input_feature_dim: torch.Size([40*16, 90])
        edge_index = batch.edge_index # torch.Size([2, 240]) 
        edge_attr = batch.edge_attr # timesteps*num_edges,edge_input_feature_dim: torch.Size([40*240, 4])

        gga = items['gga'] # torch.Size([batch_size, 40, 512])
        gga = self.gga_embedding(gga) # [batch_size, 40, 512->64]
        gga = gga.reshape(-1, self.embed_dim) # [batch_size * 40, 64]
        x_node, x_edge, gga = self.mp1(x, edge_index.long(), edge_attr, gga) # torch.Size([640, 90]), # torch.Size([9600, 4])
        x_node, x_edge, gga = self.mp2(x_node, edge_index.long(), x_edge, gga)  # torch.Size([640, 90]), # torch.Size([9600, 4])

        # LSTM
        x_node = x_node.view(timesteps, num_nodes*batch_size, -1) # torch.Size([40, batchsize*16, 90])
        # [40, batch_size, 16*output_dim] -> [40, batch_size, lstm_output_dim]
        x_node, (node_h_n, c_n) = self.node_lstm(x_node) 
        # node_hn torch.Size([1, batchsize*16, 90])

        node_h_n = node_h_n.reshape(batch_size, num_nodes, -1) # node_h_n is the last prediction
        h = node_h_n[torch.arange(node_h_n.size(0)), items['follower_ids'], :]
        h = self.regression(h)
        h = 4.0 * torch.sigmoid(h) + 1.0

        return h


