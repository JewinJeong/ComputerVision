import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))

## sigmoid를 직접 구현
hypothesis = 1 / (1 + torch.exp(-(torch.matmul(x_train,W) + b)))
# hypothesis 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

print(hypothesis)
print(hypothesis.shape)

## sigmoid 함수를 통한 구현
print('1/(1+e^{-1}) equals: ', torch.sigmoid(torch.FloatTensor([1])))

hypothesis = torch.sigmoid(x_train.matmul(W)+b)

print(hypothesis)
print(hypothesis.shape)

## cost function 구현
losses = -(y_train * torch.log(hypothesis) + (1 - y_train)*torch.log(1-hypothesis))
print(losses)

cost = losses.mean()
print(cost)

##위의 것들을 통틀어서 binary cross entropy를 이용해 구현할 수 있다.

F.binary_cross_entropy(hypothesis, y_train)
print(hypothesis)