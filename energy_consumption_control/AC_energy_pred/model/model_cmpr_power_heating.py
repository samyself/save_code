import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

import config


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=[6, 4, 2]):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.out_linear = nn.Linear(hidden_dim[2], 1)

        self.bn_1 = torch.nn.BatchNorm1d(hidden_dim[0])
        self.bn_0 = torch.nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x1 = self.bn_0(x)
        x1 = self.linear1(x1)
        x1 = torch.relu(x1)
        x1 = self.bn_1(x1)
        x2 = self.linear2(x1)
        x2 = torch.relu(x2)
        x3 = self.linear3(x2)
        out = self.out_linear(x3)

        return out
