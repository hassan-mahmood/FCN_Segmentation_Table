import torch
import os
import cv2
import numpy as np
from PIL import Image
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

    def convert_np_to_PIL(self,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    # def convert_to_label(self,imgpath):
    #     im=cv2.imread(imgpath)
    #
    #     h,w,d=im.shape
    #     label=np.zeros((h,w))
    #     #because it is BGR so we have to see where the red color is
    #     label[np.where((im == [0, 0, 255]).all(axis=2))] = [255]
    #     #label=self.convert_np_to_PIL(label)
    #     return label


    def __getitem__(self, idx):
        image=self.convert_np_to_PIL(cv2.imread(os.path.join(self.imagespath,self.images[idx])))
        label=os.path.join(self.labelspath,self.labels[idx])

        label=Image.fromarray(np.uint8(label))
        #image=torch.tensor(image)
        #following rollaxis will convert r g b c to c r g b

        image = self.transforms(image)
        label=self.transforms(label)

        return image,label



