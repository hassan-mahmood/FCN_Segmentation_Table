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
import tool
import torch.functional as F

parser=argparse.ArgumentParser()
parser.add_argument('--traindir',default="dataset/",help="Default directory which contains images and labels for training")
parser.add_argument('--valdir',default="dataset/",help="Default directory which contains images and labels for evaluation")
parser.add_argument('--checkpoint',default='None',help="If you want to load the model from a checkpoint")
parser.add_argument('--lr',default="0.0001",help="learning rate to update the parameters")
parser.add_argument('--epochs',default="100",help="number of epochs to train the model to.")
parser.add_argument('--classes',default="2",help="number of classes to segment")
parser.add_argument('--cuda',default="false",help="Set to true if want to run the model on GPU")



#ref: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
def cross_entropy2d(input, target,cuda):

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
    if(cuda):
        target=target.type(torch.cuda.LongTensor)
    else:
        target = target.type(torch.LongTensor)

    loss = nll_loss(log_p, target)
    if(cuda):
        loss=loss.type(torch.cuda.FloatTensor)
    else:
        loss = loss.type(torch.FloatTensor)
    # if size_average:
    #     loss = loss.item()/torch.sum(mask).item()
    return loss



if __name__=="__main__":

    args = parser.parse_args()

    cuda=args.cuda
    if args.cuda.lower() in ('yes', 'true', 't', 'y', '1'):
        cuda=True
    else:
        cuda=False

    starting_epoch = 0
    if(not os.path.exists('weights')):
        os.mkdir('weights')
    else:
        starting_epoch=len(os.listdir('weights'))+1

    mytransforms = torchvision.transforms.Compose([torchvision.transforms.Resize((1024, 1024)), torchvision.transforms.ToTensor()])

    train_dataset=dataset(args.traindir,mytransforms)
    val_dataset=dataset(args.valdir,mytransforms)

    train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=2)
    test_dataloader=DataLoader(val_dataset,batch_size=1,num_workers=2)

    #we are just considering buildings and background so 2 classes
    no_of_classes=int(args.classes)
    net=FCN8(no_of_classes)

    if(args.checkpoint!="None"):
        net=torch.load(args.checkpoint)

    if(cuda):
        net.cuda()

    optimizer=optim.SGD(net.parameters(),lr=float(args.lr),momentum=0.9)
    criterion = nn.BCELoss()
    softmax=nn.Softmax()
    epochs=int(args.epochs)
    print('Starting training')
    for epoch in range(starting_epoch,epochs):
        running_loss=0.0
        iteration_size=0
        #setting net/model to training mode
        net.eval()
        for i_batch,batch in tqdm(enumerate(train_dataloader)):
            image,label=batch

            if (cuda):
                image.cuda()
                label.cuda()
                image = image.type(torch.cuda.FloatTensor)
                label = label.type(torch.cuda.FloatTensor)
            else:
                image = image.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)

            optimizer.zero_grad()
            output=net(image)

            loss=cross_entropy2d(output,label,cuda)

            loss.backward()
            running_loss+=loss.item()
            optimizer.step()
            iteration_size+=1

        print("epoch {}, loss: {}".format(epoch, running_loss / iteration_size))
        torch.save(net, os.path.join('weights', str(epoch) + '.pt'))

        print('Validation:\n')
        net.eval()
        label_trues, label_preds = [], []
        for i_batch,batch in tqdm(enumerate(train_dataloader)):
            image,label=batch

            if (cuda):
                image.cuda()
                label.cuda()
                image = image.type(torch.cuda.FloatTensor)
                label = label.type(torch.cuda.FloatTensor)
            else:
                image = image.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)

            output=net(image)
            output = output.data.max(1)[1].squeeze_(1).squeeze_(0)
            if cuda:
                label = label.cpu()
                output = output.cpu()

            label_trues.append(label.numpy())
            label_preds.append(output.numpy())

        metrics = tool.accuracy_score(label_trues, label_preds)
        metrics = np.array(metrics)
        metrics *= 100
        print('''\Mean IU: {0}'''.format(*metrics))








