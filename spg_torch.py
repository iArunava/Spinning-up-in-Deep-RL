import numpy as np
import torch

from torch import nn, optim

class SPG(nn.Module):
    def __init__(self, obs_size, action_size, hid_layers):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hid_layers = hid_layers
        self.fc1 = nn.Linear(self.obs_size, self.hid_layers[0])
        self.fc2 = nn.Linear(self.hid_layers[0], self.action_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Flatten the observation
        x = x.view(1, x.shape[0])

        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        print (x)
        return x

def train_with_torch(FLAGS, obs_size, action_size):
    # Network Hyperparameters
    layers = 2
    hneurons = [32]
    epochs = FLAGS.epochs
    batch_size = 5000
    lr = 1e-2
    out_act = None

    graph_path = FLAGS.graph_path
    if graph_path[0] != '/':
        graph_path += '/'

    # Build the network
    module = SPG(obs_size, action_size, hneurons)

    return

def test_with_torch(FLAGS):
    return
