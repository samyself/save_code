import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_dim,out_dim, hidden_dim=[256, 128]):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        # self.linear3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.out_linear_1 = nn.Linear(hidden_dim[1], out_dim[0])
        self.out_linear_2 = nn.Linear(hidden_dim[1], out_dim[1])


    def forward(self, x):
        x1 = self.linear1(x)
        x1 = torch.relu(x1)

        x2 = self.linear2(x1)


        out_1 = self.out_linear_1(x2)
        out_2 = self.out_linear_2(x2)
        return out_1,out_2
