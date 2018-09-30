from FCN8.FCN8 import *
import torch
import torch.nn as nn


tensor=torch.zeros(1,3,224,224)
net=FCN8(5)
net(tensor)
#tensor=torch.zeros(1,512,14,14)






# net=nn.Sequential(nn.ConvTranspose2d(5, 5, 4, stride=2, bias=False),
#                   nn.ConvTranspose2d(5, 5, 4, stride=2, bias=False),
#                   nn.ConvTranspose2d(5, 5, 5, stride=1, bias=False))
# net=nn.Sequential(nn.ConvTranspose2d(5, 5, 4, stride=2, bias=False),
#                   nn.ConvTranspose2d(5, 5, 8, stride=2, bias=False))
# 1 1 to 14 14 net=nn.Sequential(nn.ConvTranspose2d(5, 5, (14,14), stride=(14,14), bias=False))


