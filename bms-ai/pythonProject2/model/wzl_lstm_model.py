import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
# from common_func import init_weights_xavier_uniform
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm_1 = nn.LSTM(7, 64, batch_first=True)
        self.lstm_2 = nn.LSTM(64, 32, batch_first=True)
        self.lstm_3 = nn.LSTM(32, 16, batch_first=True)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def forward(self, x):        
        x,_ = self.lstm_1(x) #[bs, 300, 64]

        x,_ = self.lstm_2(x) #[bs, 300, 32]

        x,_ = self.lstm_3(x) #[bs, 300, 16]

        x = x[:, -2:, :].contiguous().view(-1, 2, 16)

        x = self.relu(self.fc1(x))
        
        x = self.fc2(x)

        return x
        
# class LSTMModel(nn.Module):
#     def __init__(self):
#         super(LSTMModel, self).__init__()
#         self.embed_size = 128
#         self.hidden_size = 256
#         self.scale = 0.02
#         self.sparsity_threshold = 0.01
        
#         self.embedding = nn.Linear(300, self.embed_size)
#         self.layernorm = nn.LayerNorm(self.embed_size)
#         self.layernorm1 = nn.LayerNorm(self.embed_size)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Sequential(
#             nn.Linear(self.embed_size, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_size, self.embed_size)
#         )
#         self.last_l = nn.Linear(7, 1)
#         self.output = nn.Linear(self.embed_size, 2)
#         self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
#         self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

#         self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

#         self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
#     def texfilter(self, x):
#         B, N, _ = x.shape
#         o1_real = torch.zeros([B, N // 2 + 1, self.embed_size],
#                               device=x.device)
#         o1_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
#                               device=x.device)

#         o2_real = torch.zeros([B, N // 2 + 1, self.embed_size],
#                               device=x.device)
#         o2_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
#                               device=x.device)

#         o1_real = F.relu(
#             torch.einsum('bid,d->bid', x.real, self.w[0]) - \
#             torch.einsum('bid,d->bid', x.imag, self.w[1]) + \
#             self.rb1
#         )
#         o1_imag = F.relu(
#             torch.einsum('bid,d->bid', x.imag, self.w[0]) + \
#             torch.einsum('bid,d->bid', x.real, self.w[1]) + \
#             self.ib1
#         )
#         o2_real = (
#                 torch.einsum('bid,d->bid', o1_real, self.w1[0]) - \
#                 torch.einsum('bid,d->bid', o1_imag, self.w1[1]) + \
#                 self.rb2
#         )
#         o2_imag = (
#                 torch.einsum('bid,d->bid', o1_imag, self.w1[0]) + \
#                 torch.einsum('bid,d->bid', o1_real, self.w1[1]) + \
#                 self.ib2
#         )
#         y = torch.stack([o2_real, o2_imag], dim=-1)
#         y = F.softshrink(y, lambd=self.sparsity_threshold)
#         y = torch.view_as_complex(y)
#         return y

#     def forward(self, x):  
#         B, L, N = x.shape
#         x = x.permute(0, 2, 1)
#         x = self.embedding(x)  # B, N, D

#         x = self.layernorm(x)
#         x = torch.fft.rfft(x, dim=1, norm='ortho')

#         weight = self.texfilter(x)
        
#         x = x * weight
#         x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")

#         x = self.layernorm1(x)

#         x = self.dropout(x)

#         x = self.fc(x)

#         x = self.output(x)
#         x = x.permute(0, 2, 1)
#         x = self.last_l(x)
#         return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
if __name__ == "__main__":
    model = LSTMModel()
    data = torch.rand((8, 60, 7))
    model(data)
    sum = count_parameters(model)
    print(sum)