from FCN8.FCN8 import *
import torch
import torch.nn as nn
import torch.functional as F
from Buildings_Dataset.dataset import *
from torch.utils.data import DataLoader
from FCN8.FCN8 import *
import torch.optim as optim
import argparse
from tqdm import tqdm

parser=argparse.ArgumentParser()
parser.add_argument('--datadir',default="dataset/",help="Default directory which contains images and labels")



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



if __name__=="__main__":
    args = parser.parse_args()

    map_dataset=dataset(args.datadir)

    dataloader=DataLoader(map_dataset,batch_size=1,shuffle=True,num_workers=2)

    #we are just considering buildings and background so 2 classes
    no_of_classes=2
    net=FCN8(no_of_classes)
    net.cuda()

    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

    epochs=100
    print('Starting training')
    for epoch in range(epochs):
        running_loss=0.0
        #setting net/model to training mode
        net.train()
        for i_batch,batch in tqdm(enumerate(dataloader)):
            image,label=batch
            image.cuda()
            label.cuda()

            optimizer.zero_grad()
            output=net(image)
            loss=cross_entropy2d(output,label)
            loss.backward()
            running_loss+=loss.item()
            optimizer.step()
        print("epoch {}, loss: {}".format(epoch, running_loss/33.0))
        torch.save(net,str(epoch)+'.pt')


