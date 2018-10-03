import argparse
import os
import cv2
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--datadir',default="labels/",help="Default directory which contains labels")
parser.add_argument('--outputdir',default='output/',help="where the output labels will be placed")
parser.add_argument('--buildings-color', nargs='+', type=int,help="B G R colors of buildings in labels")

args=parser.parse_args()
print(args.buildings_color)

datadir=args.datadir
outputdir=args.outputdir
buildings_color=args.buildings_color

if(not os.path.exists(outputdir)):
    os.mkdir(outputdir)

for image in os.listdir(datadir):

    imagepath=os.path.join(datadir,image)
    im=cv2.imread(imagepath)
    h,w,d=im.shape
    label=np.zeros((h,w))
    label[np.where((im==buildings_color).all(axis=2))]=[255]

    #ext=os.path.splitext(image)[-1]
    #cv2.imwrite(image.split('_')[0]+ext,label)
    cv2.imwrite(os.path.join(outputdir,image), label)

print('Done')

