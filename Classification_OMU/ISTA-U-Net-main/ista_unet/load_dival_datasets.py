from dival.datasets import fbp_dataset
from torch import cuda
from torch._C import TracingState, device, dtype, set_num_interop_threads
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from os import path
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import sys
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
from torch.utils.data import Dataset as TorchDataset
import torch
from os import path
import glob
import matplotlib.pyplot as plt

from torchvision.transforms.transforms import Grayscale
from ISTA import dataset_dir
from ISTA import model_save_dir
import skimage.io as io
from PIL import Image
import cv2
import natsort
import os
    
class RandomAccessTorchDataset(TorchDataset):
    def __init__(self):
        super().__init__()
        data_dir = "/image/to/path"
        real_dir = []
        ll = 0
        for idx in range(len(data_dir)):
            jpg_lst_gt = [os.path.join(data_dir, '/image/to/path/groundtruth']
            jpg_lst_LR = [os.path.join(data_dir, '/image/to/path/noiseimage']
            for i,(a,b) in enumerate(zip(jpg_lst_LR,jpg_lst_gt)):
                a_img = cv2.imread(a,0)
                b_img = cv2.imread(b,0)
                ll += 1
                real_dir.append([a,b])
        self.data_dir = real_dir


    def __len__(self):
        # return len(self.data_dir)
        return len(self.data_dir)

    def __getitem__(self,idx):
        ira = cv2.imread(self.data_dir[idx][0],cv2.IMREAD_GRAYSCALE)
        # a   = np.zeros((512,512))
        # a[:,105:405]= sino
        ira = cv2.resize(ira,(512,512))
        # plt.imsave("/home/mars/workspace/cy_workspace/ISTA-U-Net-main/ista_unet/psnr_data/kk"+".png",a,cmap='gray')
        # ira = cv2.resize(ira,(512,512))
        ground = cv2.imread(self.data_dir[idx][1],cv2.IMREAD_GRAYSCALE)
        ground = cv2.resize(ground,(512,512))


        x=torch.tensor(ira)
        d=torch.tensor(ground)
        x=torch.unsqueeze(x,axis=0)
        d=torch.unsqueeze(d,axis=0)
        return x,d

class RandomAccessTestDataset(TorchDataset):
    def __init__(self):
        super().__init__()
        # gt = natsort.os_sorted(glob.glob("/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/input/train_jpg/success3/F_00000020/*.jpg"))
        sino = sorted(glob.glob('/home/mars/workspace/cy_workspace/Dicom/test_noise/*.png'))
        # ira = natsort.os_sorted(glob.glob("/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/input/train_jpg/success/F_00000020/*.jpg"))
        gt = natsort.os_sorted(glob.glob("/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/Registration/save/GT/44172816_original_ct/19.png"))
        ira = natsort.os_sorted(glob.glob("/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/Registration/save/CBCT/44172816_original_700_input/19.png"))
        testset = []
        for i,(s,g) in enumerate (zip(ira,gt)):
            testset.append([s,g])
        self.data_dir = testset


    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self,idx):
        # sino = cv2.imread(self.data_dir[0][0][idx])
        # ground = cv2.imread(self.data_dir[0][1][idx])
        ira = cv2.imread(self.data_dir[idx][0],cv2.IMREAD_GRAYSCALE)
        ira = cv2.resize(ira,(512,512))
        # a   = np.zeros((512,512))
        # a[:,105:405] = sino
        # a   = cv2.resize(a,(128,128))
        # ira = cv2.resize(ira,(512,512))
        ground = cv2.imread(self.data_dir[idx][1],cv2.IMREAD_GRAYSCALE)
        ground = cv2.resize(ground,(512,512))
        
       
        z=torch.tensor(ira)
        x=torch.tensor(ground)
        z=torch.unsqueeze(z,axis=0)
        x=torch.unsqueeze(x,axis=0)

        return z,x
        

