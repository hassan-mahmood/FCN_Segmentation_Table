import torch
import os
import cv2
import numpy as np

class dataset(torch.utils.data.Dataset):
    def __init__(self,dirpath):
        #this dirpath will have subfolders: images, labels
        self.dirpath=dirpath
        self.images=os.listdir(os.path.join(dirpath,'images'))
        self.labels=os.listdir(os.path.join(dirpath,'labels'))


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
        image=cv2.imread(os.path.join(self.dirpath,self.images[idx]))
        label=self.convert_to_label(os.path.join(self.dirpath,self.labels[idx]))
        return image,label



