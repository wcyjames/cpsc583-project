import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNet(nn.Module):
    def __init__(self, isize, osize, internal_sizes=[512, 256, 64], dropout=0.5, activation='relu', last_activation='relu'):
        super(FFNet, self).__init__()
        layers = []
        sizes = [isize] + internal_sizes + [osize]
        num_layers = len(sizes) - 1
        for i in range(num_layers):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            # no dropout on last layer
            if i != num_layers - 1:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.ModuleList(layers)
        self.last_layer_idx = len(layers) - 1
        self.activation = activation
        self.last_activation = last_activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # relu activate all layers  except the last layer
            if type(layer) == nn.Linear and i != self.last_layer_idx:
                if self.activation == 'relu':
                    x = F.relu(layer(x))
                elif self.activation == 'leaky_relu':
                    x = F.leaky_relu(layer(x))
                else:
                    raise ValueError('Invalid activation function')
            else:
                x = layer(x)
                if self.last_activation == 'sigmoid':
                    x = torch.sigmoid(x)
                elif self.last_activation == 'softmax':
                    x = F.softmax(x)
                elif self.last_activation == 'relu':
                    x = F.relu(x)
        return x.squeeze()