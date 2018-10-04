import numpy as np
import torch
import torch.nn as nn


output=torch.LongTensor(10,10,2).random_(0,2)
target=torch.LongTensor(10,10).random_(0,2)

output=output.numpy()
target=target.numpy()

criterion=nn.CrossEntropyLoss()
print(criterion(output,target))