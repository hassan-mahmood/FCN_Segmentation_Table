from FCN8.FCN8 import *
import torch
import torch.nn as nn
import torch.functional as F
from Buildings_Dataset.dataset import *
from torch.utils.data import DataLoader
from FCN8.FCN8 import *
import torch.optim as optim

#ref: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


map_dataset=dataset('dataset')
dataloader=DataLoader(map_dataset,batch_size=4,shuffle=True,num_workers=2)

#we are just considering buildings and background so 2 classes
no_of_classes=2
net=FCN8(no_of_classes)

optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

epochs=100

for epoch in epochs:
    running_loss=0.0
    for i_batch,batch in enumerate(dataloader):
        image,label=batch

        optimizer.zero_grad()
        output=net(image)

        loss=cross_entropy2d(output,label)
        loss.backward()
        print("iter {}, loss {}".format(iter, loss.data[0]))
        optimizer.step()

