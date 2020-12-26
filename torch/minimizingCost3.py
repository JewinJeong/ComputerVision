import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#Data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

# #Data initialize
# W = torch.zeros(1)
#
# # Setting learning Rate
# lr = 0.01
#
# nb_epoch = 10
# for epoch in range(nb_epoch + 1):
#     #H(x)
#     hypothesis = x_train * W
#
#     #Cost
#     cost = torch.mean((hypothesis - y_train)**2)
#     #Gradient
#     gradient = torch.sum((W * x_train - y_train) * x_train)
#
#     print('Epoch {:4f}/{} W: {:.3f}, cost: {:.6f}'.format(epoch, nb_epoch,W.item(), cost.item()))
#
#     # cost gradient로 H(x) 개선
#     W -= lr * gradient

## optimiziation

W = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = x_train * W

    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
    ))

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

