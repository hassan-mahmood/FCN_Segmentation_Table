
from Buildings_Dataset.dataset import *
from torch.utils.data import DataLoader
from FCN8.FCN8 import *
import argparse
from tqdm import tqdm
import torchvision
import tool

parser=argparse.ArgumentParser()
parser.add_argument('--testdir',default="dataset/",help="Default directory which contains images and labels for training")
parser.add_argument('--outputdir',default="dataset/",help="Default directory which contains images and labels for evaluation")
parser.add_argument('--checkpoint',default='None',help="If you want to load the model from a checkpoint")
parser.add_argument('--classes',default="2",help="number of classes to segment")
parser.add_argument('--cuda',default="false",help="Set to true if want to run the model on GPU")

if __name__=="__main__":

    args = parser.parse_args()
    cuda=args.cuda
    if args.cuda.lower() in ('yes', 'true', 't', 'y', '1'):
        cuda=True
    else:
        cuda=False

    if(not os.path.exists(args.outputdir)):
        os.mkdir(args.outputdir)

    mytransforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])

    test_dataset=dataset(args.testdir,mytransforms)

    test_dataloader=DataLoader(test_dataset,batch_size=1,num_workers=2)

    #we are just considering buildings and background so 2 classes
    no_of_classes=int(args.classes)
    net=FCN8(no_of_classes)

    if(args.checkpoint!="None"):
        net=torch.load(args.checkpoint)

    if(cuda):
        net.cuda()

    print('Starting Testing')
    label_trues, label_preds = [], []


    net.eval()
    for i_batch,batch in tqdm(enumerate(test_dataloader)):
        image,label=batch
        if(cuda):
            image.cuda()
            label.cuda()
            image = image.type(torch.cuda.FloatTensor)
            label=label.type(torch.cuda.FloatTensor)
        else:
            image = image.type(torch.FloatTensor)
            label=label.type(torch.FloatTensor)

        output=net(image)
        output = output.data.max(1)[1].squeeze_(1).squeeze_(0)

        if cuda:
            label=label.cpu()
            output=output.cpu()

        label_trues.append(label.numpy())
        label_preds.append(output.numpy())
        tool.labelTopng(output,os.path.join(args.outputdir,str(i_batch)+'.png'))


    metrics = tool.accuracy_score(label_trues, label_preds)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}'''.format(*metrics))
