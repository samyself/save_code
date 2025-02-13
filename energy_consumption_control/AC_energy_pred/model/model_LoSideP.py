import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

import config_all as config


class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, 2)

        self.bn1 = torch.nn.BatchNorm1d(16)
        self.bn2 = torch.nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.bn1(torch.relu(self.linear1(x)))

        x = self.bn2(torch.relu(self.linear2(x)))

        x = self.linear3(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)