import numpy as np
import torch
# a =1.00
#
# c = np.sqrt(np.exp(a))
#
# print(c)

a = np.array([110 , 308 , 85, 306, 365, 1237, 22, 1196])
# # # b =np.array(5855230)
c = np.sqrt(np.exp(a))
print(a / 15119)

# x = np.array([1, 2, 2, 1, 2, 1, 2, 2, 2, 1])
# y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
# c = torch.bincount(x * len(np.unique(y)) + y)