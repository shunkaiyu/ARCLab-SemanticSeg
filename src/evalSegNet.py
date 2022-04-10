'''
Semantic Segmentation Network Evaluation
(Work-In-Progress)
Stand-alone utility to evaluate segmentation predictions using a trained model.
'''

# torch imports
import segmentation_models_pytorch as smp
import cv2

# general imports
import argparse
import os

# utility imports
from torchvision.transforms.transforms import ToTensor
import json

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from torch.nn.functional import one_hot
from utils import dice
from dice_loss import DiceLoss
from skimage.metrics import hausdorff_distance

import numpy as np
import matplotlib.pyplot as plt
import utils
from model.segnet import SegNet
from data.dataloaders.SegNetDataLoaderV2 import SegNetDataset

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--batchSize', default=1, type=int,
            help='Mini-batch size (default: 1)')
parser.add_argument('--bnMomentum', default=0.1, type=float,
            help='Batch Norm Momentum (default: 0.1)')
parser.add_argument('--imageSize', default=256, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--model', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the evaluated images',
            default='save_temp', type=str)
parser.add_argument('--saveTest', default='False', type=str,
            help='Saves the validation/test images if True')
parser.add_argument('--data_path',
            help='Data directory to be evaluated', default='data_temp', type=str)

#use_gpu = torch.cuda.is_available()
# GPU Check
use_gpu = torch.cuda.is_available()
curr_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(curr_device)
device = torch.device('cuda' if use_gpu else 'cpu')

def reverseOneHot(batch, key):
    '''
        Generates the segmented image from the output of a segmentation network.
        Takes a batch of numpy oneHot encoded tensors and returns a batch of
        numpy images in RGB (not BGR).
    '''

    seg = []

    # Iterate over all images in a batch
    for i in range(len(batch)):
        vec = batch[i]
        idxs = vec

        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k
            segSingle[mask] = rgb

        segMask = np.expand_dims(segSingle, axis=0)
        
        seg.append(segMask)
    
    seg = np.concatenate(seg)

    return seg

def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.saveTest == 'True':
        args.saveTest = True
    elif args.saveTest == 'False':
        args.saveTest = False

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cudnn.benchmark = True

    data_transform = transforms.Compose([
            # transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.Resize((480, 854), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    # Data Loading
    #data_dir = "../src/data/datasets/cholec_12_3"
    data_dir = args.data_path
    # json path for class definitions
    json_path = "../src/data/classes/cholecSegClasses.json"

    image_dataset = SegNetDataset(os.path.join(data_dir,'test'), data_transform, json_path, 'test')
    print(len(image_dataset))
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=args.batchSize,
                                             shuffle=True,
                                             num_workers=args.workers)

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_dataset.classes
    key = utils.disentangleKey(classes)
    num_classes = len(key)

    # Initialize the model
    # TODO: match model initialization code with trainSegNet.py (i.e. model = UNet, ResNetUNet, etc.)
    # model = SegNet(args.bnMomentum, num_classes)
    model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            in_channels=3,
            classes=num_classes
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Load the saved model
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    print(model)

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Initialize an evaluation Object
    evaluator = utils.Evaluate(key, use_gpu)

    # Evaulate on validation/test set
    print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
    val_loss, val_dice_coeff, val_haus_dist = validate(dataloader, model, criterion, key, evaluator)
    print(f"Loss={val_loss}, Avg. DC={val_dice_coeff}, Avg. HD={val_haus_dist}")

    # Calculate the metrics
    print('>>>>>>>>>>>>>>>>>> Evaluating the Metrics <<<<<<<<<<<<<<<<<')
    IoU = evaluator.getIoU()
    print('Mean IoU: {}, Class-wise IoU: {}'.format(torch.mean(IoU), IoU))
    PRF1 = evaluator.getPRF1()
    precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]
    print('Mean Precision: {}, Class-wise Precision: {}'.format(torch.mean(precision), precision))
    print('Mean Recall: {}, Class-wise Recall: {}'.format(torch.mean(recall), recall))
    print('Mean F1: {}, Class-wise F1: {}'.format(torch.mean(F1), F1))

def validate(val_loader, model, criterion, key, evaluator):
    '''
        Run evaluation
    '''
    total_val_loss = 0
    total_dice_coeff = 0
    total_haus_dist = 0
    avg_dice_coeff = 0
    avg_haus_dist = 0
    total_samples = args.batchSize

    val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
    # Switch to evaluate mode
    model.eval()

    for i, (img, gt, label) in enumerate(val_loader):

        # Process the network inputs and outputs
        img = utils.normalize(img, torch.Tensor([0.337, 0.212, 0.182]), torch.Tensor([0.278, 0.218, 0.185]))
        #img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))
        oneHotGT = one_hot(label, len(key)).permute(0, 3, 1, 2)
	
        img, label = Variable(img), Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()
            oneHotGT = oneHotGT.cuda()

        # Compute output
        seg = model(img)
        #print("-----------")
        #print(seg.shape)
        seg = TF.resize(seg, [480, 854], interpolation=Image.NEAREST)

        #seg = label.clone().detach()
        loss = criterion(seg, label)
        
        total_val_loss += loss.mean().item()

        evaluator.addBatch(seg, oneHotGT, args)

        seg = torch.argmax(seg, dim=1)

        # Dice Coefficient and Hausdorff Distance Metrics every 10 epochs
        #if (epoch+1) == 1 or (epoch+1) % 1 == 0:
        if True:
            for seg_im, label_im in zip(seg, label): # iterate over each image in the batch
                seg_im, label_im = one_hot(seg_im, len(key)), one_hot(label_im, len(key))
                seg_im, label_im = seg_im.cpu(), label_im.cpu()
                seg_im, label_im = seg_im.permute(2, 0, 1), label_im.permute(2, 0, 1)
                total_dice_coeff += dice(seg_im.data, label_im.data)

                #if args.dataset == "synapse":
                #    seg_im, label_im = seg_im[:21], label_im[:21]

                for seg_slice, label_slice in zip(seg_im, label_im): # iterate over each image slice
                    seg_slice, label_slice = seg_slice.numpy(), label_slice.numpy()
                    total_haus_dist += 1000 if hausdorff_distance(seg_slice, label_slice) == np.inf else hausdorff_distance(seg_slice, label_slice)
            
            avg_dice_coeff = total_dice_coeff / total_samples
            avg_haus_dist = total_haus_dist / total_samples

            total_samples += args.batchSize

            val_loop.set_postfix(avg_loss = total_val_loss / (i + 1), avg_dice = avg_dice_coeff, avg_haus_dist = avg_haus_dist) # avg dice coefficient and avg hausdorff distance per image
        else:
            val_loop.set_postfix(avg_loss = total_val_loss / (i + 1))
        
        val_loop.set_description(f"i = [{i + 1}]")

        #print('[%d/%d] Loss: %.4f' % (i, len(val_loader)-1, loss.mean().item()))

        #utils.displaySamples(img, seg, gt, use_gpu, key, args.saveTest, 0, i, args.save_dir)
        
    return total_val_loss/len(val_loop), avg_dice_coeff, avg_haus_dist

        #evaluator.addBatch(seg, oneHotGT)

if __name__ == '__main__':
    main()
