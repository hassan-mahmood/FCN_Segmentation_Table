import torch
import os

class dataset(torch.utils.data.Dataset):
    def __init__(self,dirpath):
        #this dirpath will have subfolders: images, labels
        self.dirpath=dirpath
        self.images=os.listdir(os.path.join(dirpath,'images'))
        self.labels=os.listdir(os.path.join(dirpath,'labels'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pass


