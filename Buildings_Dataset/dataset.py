import torch


class dataset(torch.utils.data.Dataset):
    def __init__(self,dirpath):
        #this dirpath will have all the images (train or test) along with their labeled masks
        self.dirpath=dirpath
        

