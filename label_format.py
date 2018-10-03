import argparse
import os
import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--datadir',default="labels/",help="Default directory which contains labels")
parser.add_argument('--outputdir',default='output/',help="where the output labels will be placed")


args=parser.parse_args()

datadir=args.datadir
outputdir=args.outputdir

if(not os.path.exists(outputdir)):
    os.mkdir(outputdir)

for image in os.listdir(datadir):
    image=os.path.join(datadir,image)
    im=cv2.imread(image)
    h,w,d=im.shape
    label=np.zeros((h,w))
    label[np.where((im==[255,0,0]).all(axis=2))]=[255]

    #ext=os.path.splitext(image)[-1]
    #cv2.imwrite(image.split('_')[0]+ext,label)
    cv2.imwrite(os.path.join(outputdir,image), label)

print('Done')

