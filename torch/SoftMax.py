import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

z = torch.rand(3,5,requires_grad=True)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5,(3,)).long()
print(y)

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1),1)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
F.nll_loss(F.log_softmax(z, dim=1), y)
F.cross_entropy(z,y)
