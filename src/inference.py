'''
Semantic Segmentation Network Inference
Stand-alone utility to generate segmentation mask for 1 surgical image as input

Note: If you get error with DeepLabV3+, please see bottom of inference.py
'''

# torch imports
import torch
import deeplabv3
from deeplabv3 import DeepLabV3Plus
import segmentation_models_pytorch as smp
import cv2

# general imports
import argparse
import os

# utility imports
import numpy as np
from torchvision.transforms.transforms import ToTensor
from PIL import Image
import json
from tqdm import tqdm

# model imports
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='Semantic Segmentation Inference Parameters')

# INFERENCE PARAMETERS
parser.add_argument('--imdir_path', default=None, type=str, help="Path to a directory of images (default: None)")
parser.add_argument('--model_path', default=None, type=str, help='Path to latest model checkpoint (default: None)')
parser.add_argument('--classes_path', default=None, type=str, help="Path to JSON file with Class Keys (default: None)")

# SAVING PARAMETERS
parser.add_argument('--save_dir', type=str, help='The directory used to save the inference image')

def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = color_array

    return dKey

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

def normalize(batch, mean, std):
    '''
        Normalizes a batch of images, provided the per-channel mean and
        standard deviation.
    '''

    mean.unsqueeze_(1).unsqueeze_(1)
    std.unsqueeze_(1).unsqueeze_(1)
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = img.sub(mean).div(std).unsqueeze(0)

        if 'concat' in locals():
            concat = torch.cat((concat, img), 0)
        else:
            concat = img

    return concat

def main():
    args, unparsed = parser.parse_known_args()

    # GPU Check
    use_gpu = torch.cuda.is_available()
    curr_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(curr_device)
    device = torch.device('cuda' if use_gpu else 'cpu')

    print("CUDA AVAILABLE:", use_gpu, flush=True)
    print("DEVICE NAME:", device_name, flush=True)

    image_mean = [0.337, 0.212, 0.182] # mean [R, G, B]
    image_std = [0.278, 0.218, 0.185] # standard deviation [R, G, B]

    classes = json.load(open(args.classes_path))["classes"]
    key = disentangleKey(classes)
    num_classes = len(key)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        in_channels=3,
        classes=num_classes
    )
    #model = DeepLabV3Plus(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=num_classes)
    model.to(device)

    checkpoint = torch.load("smp_DeepLabV3+_cholec_bs12lr0.001e50_checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    #for root, dirs, files in os.walk(args.imdir_path, topdown=False):
    #    for name in tqdm(files, total=len(files)):
    #        image_path = os.path.join(root, name)
    for i in range(0,1):
    	for j in range(0,1):
    	    image_path = args.imdir_path
    	    name = os.path.split(image_path)
    	    name = name[1]
    	    image = Image.open(image_path)
    	    to_tensor = ToTensor()
    	    im = to_tensor(image)
    	    im = TF.resize(im, [256, 256], interpolation=TF.InterpolationMode.BILINEAR)
    	    im = torch.unsqueeze(im, 0)
    	    im = normalize(im, torch.Tensor(image_mean), torch.Tensor(image_std))
    	    im = im.to(device)
    	    seg = model(im)
    	    seg = torch.argmax(seg, dim=1)
    	    seg = seg.cpu()
    	    seg = seg.data.numpy()
    	    seg = reverseOneHot(seg, key)
    	    seg = np.squeeze(seg[0]).astype(np.uint8)
    	    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    	    print(seg.shape)
    	    #seg = TF.resize(seg, [480, 854], interpolation=Image.NEAREST)
    	    seg = cv2.resize(seg, (854, 480), 0, 0, interpolation = cv2.INTER_NEAREST)
    	    print(seg.shape)
    	    seg_save_path = os.path.join(args.save_dir, f"{name}")
    	    image_save_path = os.path.join(args.save_dir, f"{name}_input.png")
    	    if not os.path.isdir(args.save_dir):
    	        os.mkdir(args.save_dir)
    	    cv2.imwrite(seg_save_path, seg)
    	    image.save(image_save_path)
    print("Inference Complete!")

if __name__ == "__main__":
    main()

'''
If you get an error with DeepLabV3+ such as:
RuntimeError: Sizes of tensors must match except in dimension 2. Got 270 and 268 (The offending index is 0)

Please add these lines to "/home/username/miniconda3/envs/torch/lib/python3.8/site-packages/segmentation_models_pytorch/deeplabv3/decoder.py"
prior to line 102:

max_dim_2 = max(high_res_features.size(2), aspp_features.size(2))
aspp_features = F.pad(aspp_features, (0, 0, 0, max_dim_2-aspp_features.size(2), 0, 0), "constant", 0)

Overall, the forward pass for DeepLabV3PlusDecoder() should look like:

def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        max_dim_2 = max(high_res_features.size(2), aspp_features.size(2))
        aspp_features = F.pad(aspp_features, (0, 0, 0, max_dim_2-aspp_features.size(2), 0, 0), "constant", 0)
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features
'''
