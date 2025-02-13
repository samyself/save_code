import torch
import torch.nn as nn
import torch.nn.init as init
from torch import dtype

import config_all

class inter1(nn.Module):
    def __init__(self):
        super().__init__()
        self.k0 = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.k1 = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.k2 = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.k3 = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.k4 = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.k5 = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x):
        return self.k0 + self.k1 * x + self.k2 * x ** 2 + self.k3 * x ** 3 + self.k4 * x ** 4 + self.k5 * x ** 5

