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
                 phi_output_size=None, gamma_output_size=None,
                 dropout=0.5, verbose=False):
        super().__init__(aggr=aggr)
        if node_features == 0:
            raise ValueError('node_features must be specified')
        if internal_sizes is None:
            # single hidden layer of size node_features
            internal_sizes = [node_features]
        # edge update input size is equal to the $x_j$, $x_i$, and $e_{j,i}$
        phi_input_size = node_features + node_features + edge_features
        # make the output of phi the same size as a node embedding
        if phi_output_size is None or phi_output_size == 0:
            phi_output_size = edge_features
        self.phi_output_size = phi_output_size
        # gamma's input is $x_i^{k-1}$ and $\bigoplus$
        gamma_input_size = node_features + phi_output_size
        # output of gamma is the tuple (node_features, edge_features)
        if gamma_output_size is None or gamma_output_size == 0:
            gamma_output_size = node_features
        self.gamma_output_size = gamma_output_size

        self.verbose = verbose

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
    
    def __repr__(self):
        res = f"phi: {self.phi}\n"
        res += f"gamma: {self.gamma}\n"
        return res

    def forward(self, x, edge_index, edge_attr):
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
        node_weights, edge_weights = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        return node_weights, edge_weights

    def message(self, x_i, x_j, edge_attr, x):
        '''
        edge update function:
          combine $x_i^{k-1}$, $x_j^{k-1}$, $e_{j,i}$ via $\phi^{(k)}$
          batch dimension (ie. the first dimension) is $k$ where $k-1$ is the previous layer
            x_i: (num_edges, num_node_features)
            x_j: (num_edges, num_node_features)
            edge_attr: (num_edges, num_edge_features)
            x: (num_nodes, num_node_features)
        return: phi which is aggregated into $\bigoplus$ by pytorch geometric
        '''
        aggregation_data = torch.cat((x_i, x_j, edge_attr), dim=1)
        if self.verbose:
            print(f"sending: {aggregation_data.shape} through phi: {self.phi}")
        phi = self.phi(aggregation_data)
        if self.verbose:
            print(f"phi output: {phi.shape}")
        return phi

    def aggregate(self, inputs, index, ptr, dim_size):
        agg = super().aggregate(inputs, index, ptr, dim_size)
        # return the aggregation and phi
        return agg, inputs

    def update(self, agg_out, x):
        '''
        node update function:
            combine $x_i^(k-1)$, $\bigoplus$ (ie. agg_out) into the node representation for layer $k$
            agg_out is the output of the aggregation function
            combines the aggregated edge update with the node embedding and returns the updated node and edge embeddings
        '''
        agg, phi = agg_out
        aggregation_data = torch.cat((x, agg), dim=1)
        if self.verbose:
            print(f"sending: {aggregation_data.shape} through gamma: {self.gamma}")
        gamma = self.gamma(aggregation_data)
        if self.verbose:
            print(f"gamma output: {gamma.shape}, phi output: {phi.shape}")
        return gamma, phi


class TemporalEmbeddedMPGNN(torch.nn.Module):
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
                 num_nodes=None, osize=1, gnn_internal_sizes=[64], ffnet_internal_sizes=[512, 256, 64], aggregate_over=[], dropout=0.0, verbose=True,
                 activation='relu', last_activation='relu'):
        super().__init__()
        aggregation_feature_options = ['node', 'edge']
        self.gga_dim = gga_dim
        self.temporal_embed_dim = temporal_embed_dim
        self.aggregate_over = aggregate_over
        if not aggregate_over:
            raise ValueError('aggregate_over must be specified')
        for a in aggregate_over:
            if a not in aggregation_feature_options:
                raise ValueError(f"aggregate_over must be one of {aggregation_feature_options}")
        self.verbose = verbose
        self.mp1 = MessagePassingLayer(
            node_features=temporal_embed_dim,
            edge_features=temporal_embed_dim,
            internal_sizes=gnn_internal_sizes,
            dropout=dropout,
            verbose=verbose
        )
        self.mp2 = MessagePassingLayer(
            node_features=temporal_embed_dim,
            edge_features=temporal_embed_dim,
            internal_sizes=gnn_internal_sizes,
            # $$$ do we want this? decreased the dimensionality entering the node update
            phi_output_size=intermediate_edge_features,
            gamma_output_size=intermediate_node_features,
            dropout=dropout,
            verbose=verbose
        ) 
        node_feature_size = 0
        if 'node' in aggregate_over:
            node_feature_size += num_nodes * self.mp2.gamma_output_size 
        if 'edge' in aggregate_over:
            node_feature_size += num_nodes * (num_nodes-1) * self.mp2.phi_output_size
        ffnet_input_size = node_feature_size
        
        self.temporal_node_embedding = torch.nn.Linear(node_features, self.temporal_embed_dim)
        self.temporal_edge_embedding = torch.nn.Linear(edge_features, self.temporal_embed_dim)

        if self.gga_dim is not None:
            gga_features = self.gga_dim[1]
            self.gga_temporal_embedding = torch.nn.Linear(gga_features, self.temporal_embed_dim)
            ffnet_input_size += self.temporal_embed_dim * timesteps
        
        self.ff = FFNet(
            isize=ffnet_input_size,
            internal_sizes=ffnet_internal_sizes,
            osize=osize,
            dropout=dropout,
            activation=activation,
            last_activation=last_activation
        )

    def __repr__(self):
        res = f"mp1: {self.mp1}\n"
        res += f"mp2: {self.mp2}\n"
        res += f"ff: {self.ff}\n"
        return res

    def forward(self, items):
        # node embedding over time
        x = items['x']
        batch_size, timesteps, features = x.shape
        x_embed = self.temporal_node_embedding(x.reshape(batch_size*timesteps, features)).reshape(batch_size, timesteps, -1)

        # edge embedding over time
        e = items['edge_attr']
        batch_size, timesteps, features = e.shape
        edge_embed = self.temporal_edge_embedding(e.reshape(batch_size*timesteps, features)).reshape(batch_size, timesteps, -1)

        graphs = []
        for i in range(x_embed.shape[0]):
            # requires an un-permutation of the time-first permutation in simple_graph_dataset
            # node_x shape is (timesteps, temporal_embed_dim)
            node_x = x_embed[i]
            edge_index = items['edge_index'][i]
            # edge_attr shape is (timesteps, temporal_embed_dim)
            edge_attr = edge_embed[i]
            if self.verbose:
                print(f"building graph Data object from sample: {i}")
                print(f"  x: {node_x.shape}")
                print(f"  edge_index: {edge_index.shape}")
                print(f"  edge_attr: {edge_attr.shape}")
            graphs.append(Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr))
        batch = Batch.from_data_list(graphs)
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        x_node, x_edge = self.mp1(x, edge_index.long(), edge_attr)
        x_node, x_edge = self.mp2(x_node, edge_index.long(), x_edge)

        # reshape output by graph (ie. batch dimension) and send it through an MLP
        to_cat = []
        if 'node' in self.aggregate_over:
            x_node_by_graph = self.attrs_by_graph(len(graphs), x_node)
            x_node_by_graph = x_node_by_graph.flatten(1)
            to_cat.append(x_node_by_graph)
        if 'edge' in self.aggregate_over:
            x_edge_by_graph = self.attrs_by_graph(len(graphs), x_edge)
            x_edge_by_graph = x_edge_by_graph.flatten(1)
            to_cat.append(x_edge_by_graph)

        # gga embedding over time
        if self.gga_dim is not None:
            gga = items['gga']
            batch_size, timesteps, features = gga.shape
            gga_embed = self.gga_temporal_embedding(gga.reshape(batch_size*timesteps, features)).reshape(batch_size, timesteps, -1)
            to_cat.append(gga_embed.flatten(1))

        if len(to_cat) > 1:
            per_graph_features = torch.cat(to_cat, 1) 
        else:
            per_graph_features = to_cat[0]

        # final output, w/ the (optional) global graph attribute
        u = self.ff(per_graph_features)

        return u
    
    def attrs_by_graph(self, num_graphs, attrs):
        '''
        in: [edge|graph] attributes of a batched graph in dimension (num_all_[edges|nodes], num_features)
        returns: (num_graphs, num_[edges|nodes], num_features)
        '''
        return attrs.reshape((num_graphs, -1, attrs.shape[-1]))



