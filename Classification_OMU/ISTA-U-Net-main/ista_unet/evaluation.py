import enum
import numpy as np
import matplotlib.pyplot as plt
import sys, os, torch
import torch.nn as nn
import torch.optim as optim
import itertools

from tqdm import tqdm
# from ista_unet import *
from models import ista_unet
from evaluate import *
from dival.measure import PSNR, SSIM
from piq import psnr
import torch
from torch.utils.data.dataloader import DataLoader
from dival.util.plot import plot_images
import dival
import pdb
import cv2
import glob
import natsort
import nibabel as nib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# loaders_dataset = RandomAccessTestDataset()
# loaders = DataLoader(loaders_dataset,batch_size=1,num_workers=0,shuffle=False)
# guid = '42c09cf4-3d56-4e99-ada7-73a5cbf9e09f' # nose_annotation
guid = 'a3d5a1af-e159-43e4-ab89-473fc6af0ff5' # all
currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
model, config_dict = load_ista_unet_model(guid = guid, 
                             dataset = 'ct',
                             model_save_dir= '/'.join(currentpath[0:-2])+"/ISTA-U-Net-main/output_dir/",
                             return_config_dict = True)

model.to(device)
# save_dir = '/'.join(currentpath[0:-2])+"/input/train_jpg/cbct_deno/" #internal
# m = natsort.os_sorted(glob.glob('/'.join(currentpath[0:-2])+"/input/train_jpg/success_cbct_90/*")) #internal
folds = 5
for fold in range(folds):
    save_dir = '/'.join(currentpath[0:-2])+"/input_external/external_jpg_cbct_90_de"+str(fold)+"/" # external
    m = natsort.os_sorted(glob.glob('/'.join(currentpath[0:-2])+"/input_external/external_jpg_cbct_90/*")) # external
    for i,s in enumerate(m):
        img = natsort.os_sorted(glob.glob(s+'/*'))
        for j,z in enumerate(img):
            test = cv2.imread(z,cv2.IMREAD_GRAYSCALE)
            test = torch.tensor(test).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            denoised = model(test)
            denoised = np.array(denoised.squeeze().cpu().detach())
            os.makedirs(os.path.join(save_dir,s.split('/')[-1]),exist_ok=True)
            plt.imsave(save_dir+s.split('/')[-1]+'/'+z.split('/')[-1],denoised[:,:],cmap='gray')
folds = 5
for fold in range(folds):
    save_dir = '/'.join(currentpath[0:-2])+"/input/train_jpg/success_cbct_90_de"+str(fold)+"/"  # internal
    m = natsort.os_sorted(glob.glob('/'.join(currentpath[0:-2])+"/input/train_jpg/success_cbct_90/*")) #internal
    for i,s in enumerate(m):
        img = natsort.os_sorted(glob.glob(s+'/*'))
        for j,z in enumerate(img):
            test = cv2.imread(z,cv2.IMREAD_GRAYSCALE)
            test = torch.tensor(test).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            denoised = model(test)
            denoised = np.array(denoised.squeeze().cpu().detach())
            os.makedirs(os.path.join(save_dir,s.split('/')[-1]),exist_ok=True)
            plt.imsave(save_dir+s.split('/')[-1]+'/'+z.split('/')[-1],denoised[:,:],cmap='gray')

