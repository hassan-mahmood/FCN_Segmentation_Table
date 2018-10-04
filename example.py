#this file will create validation dataset from a given directory

import argparse
import os
import random
import shutil
parser=argparse.ArgumentParser()

parser.add_argument('--datadir',default='dataset/',help='Directory which contains images and labels folders')
parser.add_argument('--outputdir',default='valdataset',help='Directory where labels and images of validation dataset will be created')


if(__name__=='__main__'):
    args=parser.parse_args()
    datadir=args.datadir
    outputdir=args.outputdir

    all_images=os.listdir(os.path.join(datadir,'images'))
    all_labels=os.listdir(os.path.join(datadir,'labels'))
    val_data_len=int(len(all_images)*0.2)

    filenames=random.sample(all_images,val_data_len)
    print(filenames)
    for f in filenames:
        shutil.move(os.path.join(datadir,'images',f),os.path.join(outputdir,'images',f))
        shutil.move(os.path.join(datadir,'labels',f),os.path.join(outputdir,'labels',f))
        pass


