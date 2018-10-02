import torch
import os
import cv2
import numpy as np
from torchvision import transforms

class dataset(torch.utils.data.Dataset):
    def __init__(self,dirpath,transforms):
        #this dirpath will have subfolders: images, labels
        self.transforms=transforms
        self.imagespath=os.path.join(dirpath,'images')
        self.labelspath=os.path.join(dirpath,'labels')
        self.images=os.listdir(self.imagespath)
        self.labels=os.listdir(self.labelspath)



    def __len__(self):
        return len(self.images)


    def convert_to_label(self,imgpath):
        im=cv2.imread(imgpath)
        h,w,d=im.shape
        label=np.zeros((h,w))
        #because it is BGR so we have to see where the red color is
        label[np.where((im == [0, 0, 255]).all(axis=2))] = [255]
        normalized = cv2.normalize(label, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return normalized


    def __getitem__(self, idx):
        image=cv2.imread(os.path.join(self.imagespath,self.images[idx]))
        label=self.convert_to_label(os.path.join(self.labelspath,self.labels[idx]))
        #image=torch.tensor(image)
        #following rollaxis will convert r g b c to c r g b
        image=np.rollaxis(image,2,0)
        # image=torch.from_numpy(image).float().cuda()
        # label=torch.from_numpy(label).float().cuda()
        image = self.transforms(image)
        label=self.transforms(label)
        #image=torch.tensor(image,dtype=torch.cuda.float)
        #label = torch.tensor(label,dtype=torch.cuda.float)
        return image,label



