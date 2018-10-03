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
import torchvision
import torch.functional as F

parser=argparse.ArgumentParser()
parser.add_argument('--traindir',default="dataset/",help="Default directory which contains images and labels for training")
parser.add_argument('--testdir',default="dataset/",help="Default directory which contains images and labels for evaluation")
parser.add_argument('--checkpoint',default='None',help="If you want to load the model from a checkpoint")
parser.add_argument('--lr',default="0.0001",help="learning rate to update the parameters")
parser.add_argument('--epochs',default="100",help="number of epochs to train the model to.")
parser.add_argument('--classes',default="2",help="number of classes to segment")
#ref: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
def cross_entropy2d(input, target):

    log_softmax=nn.LogSoftmax()
    nll_loss=nn.NLLLoss()
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    target=target.type(torch.cuda.LongTensor)
    loss = nll_loss(log_p, target)
    loss=loss.type(torch.cuda.FloatTensor)
    # if size_average:
    #     loss = loss.item()/torch.sum(mask).item()
    return loss


def iou(pred, target,no_of_classes):
  ious = []
  # Ignore IoU for background class
  for cls in range(no_of_classes - 1):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection / max(union, 1))
  return ious


if __name__=="__main__":

    args = parser.parse_args()
    starting_epoch = 0
    if(not os.path.exists('weights')):
        os.mkdir('weights')
    else:
        starting_epoch=len(os.listdir('weights'))+1

    mytransforms = torchvision.transforms.Compose([torchvision.transforms.Resize((1024, 1024)), torchvision.transforms.ToTensor()])

    train_dataset=dataset(args.traindir,mytransforms)
    test_dataset=dataset(args.testdir,mytransforms)

    train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=2)
    test_dataloader=DataLoader(test_dataset,batch_size=1,num_workers=2)

    #we are just considering buildings and background so 2 classes
    no_of_classes=int(args.classes)
    net=FCN8(no_of_classes)

    if(args.checkpoint!="None"):
        net=torch.load(args.checkpoint)
    #net.cuda()

    optimizer=optim.SGD(net.parameters(),lr=float(args.lr),momentum=0.9)
    criterion = nn.BCELoss()
    softmax=nn.Softmax()
    epochs=int(args.epochs)
    print('Starting training')
    for epoch in range(starting_epoch,epochs):
        running_loss=0.0
        iteration_size=0
        #setting net/model to training mode
        net.train()
        for i_batch,batch in tqdm(enumerate(train_dataloader)):
            image,label=batch

            image.cuda()
            label.cuda()
            image = image.type(torch.cuda.FloatTensor)
            label=label.type(torch.cuda.FloatTensor)

            optimizer.zero_grad()
            output=net(image)

            loss=cross_entropy2d(output,label)
            print('Loss value:',loss.item())

            loss.backward()
            running_loss+=loss.item()
            optimizer.step()
            iteration_size+=1

        print("epoch {}, loss: {}".format(epoch, running_loss/iteration_size))
        torch.save(net,os.path.join('weights',str(epoch)+'.pt'))

