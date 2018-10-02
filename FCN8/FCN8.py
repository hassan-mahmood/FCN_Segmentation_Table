import torch
from torchvision.models import *
import torch.nn as nn


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super(Conv2dSame,self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class FCN8(nn.Module):

    def __init__(self,no_of_classes):
        super(FCN8,self).__init__()
        #we will use pretrained vgg model of pytorch
        vgg=vgg16(pretrained=True)
        basemodel = nn.Sequential(*list(vgg.children())[0])

        self.pool3=nn.Sequential(*list(basemodel.children())[:17])
        self.pool4=nn.Sequential(*list(basemodel.children())[17:24])
        self.pool5 = nn.Sequential(*list(basemodel.children())[24:])
        #now the basemodel has all vgg layers before fc layers So we will add fully convolutional layers


        self.fconv6=Conv2dSame(512,4096,7)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout2d(p=0.5)



        self.fconv7=Conv2dSame(4096,4096,1)
        self.score_fr=Conv2dSame(4096,no_of_classes,1)

        self.conv7_to_pool4=nn.ConvTranspose2d(no_of_classes, no_of_classes, (2,2), stride=(2,2), bias=False)
        self.pool4_1=Conv2dSame(512,no_of_classes,1,1)

        self.pool4_to_pool3 = nn.ConvTranspose2d(no_of_classes, no_of_classes, (2, 2), stride=(2, 2), bias=False)
        self.pool3_1 = Conv2dSame(256, no_of_classes, 1, 1)

        self.upsample_to_original=nn.ConvTranspose2d(no_of_classes,no_of_classes,(8,8),stride=(8,8),bias=False)

        #now we will upsample this fconv7 to the size of pool 4 and pool 3


    def forward(self, img):

        out_pool3=self.pool3(img)
        out_pool4=self.pool4(out_pool3)
        out = self.pool5(out_pool4)
        #above out is output from vgg

        print('pool 5 shape:',out.shape)

        out=self.dropout(self.relu(self.fconv6(out)))
        print('Pool 6 shape:',out.shape)
        out=self.dropout(self.relu(self.fconv7(out)))
        print('Pool 7 shape:',out.shape)
        out=self.score_fr(out)
        print('Pool 7 fr shape:',out.shape)


        conv7_upsampled=self.conv7_to_pool4(out)
        pool4_1=self.pool4_1(out_pool4)
        #now the conv7 and pool4 are of same height, width and depth
        out=conv7_upsampled+pool4_1


        pool4_upsampled=self.pool4_to_pool3(out)
        pool3_1=self.pool3_1(out_pool3)
        out=pool4_upsampled+pool3_1

        out=self.upsample_to_original(out)
        #print('Final output shape:',out.shape)
        return out







