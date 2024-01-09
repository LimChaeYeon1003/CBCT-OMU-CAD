import os
import sys
sys.path.append(os.path.join("../Codes_3D/"))
import random
import torch
import argparse
import numpy as np
import pandas as pd
import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor
from monai.transforms import Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine, CenterSpatialCrop, RandFlip
from monai.utils import get_seed
from glob import glob
import cv2
from pylab import rcParams
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import time
import csv
import shutil
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from matplotlib.ticker import MaxNLocator
from models import resnet
from models.model import (generate_model, load_pretrained_model, make_data_parallel, get_fine_tuning_parameters)
import albumentations
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.simplefilter("ignore", UserWarning)
from medcam import medcam
import nibabel as nib
import matplotlib.cm as cm
import imageio

def get_models():
    if args.model == "densenet121": 
        model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=3, out_channels=3)
    elif args.model == "resnet18":  
        model = resnet.generate_model(model_depth=18, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D/models/r3d18_K_200ep.pth", 'resnet', 3)
    elif args.model == "resnet34":  
        model = resnet.generate_model(model_depth=34, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D/models/r3d34_K_200ep.pth", 'resnet', 3)
    elif args.model == "resnet50":  
        model = resnet.generate_model(model_depth=50, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D/models/r3d50_K_200ep.pth", 'resnet', 3)
    if args.weight_dir is not None: 
        print(f"Load '{args.weight_dir}'...")
        try:  # single GPU model_file
            model.load_state_dict(torch.load(args.weight_dir), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(args.weight_dir)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
    return model

class OMUDataset3D(torch.utils.data.Dataset, Randomizable):
    def __init__(self, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.patients = csv.ID.unique()
        self.mode = mode
        self.transform = transform
    def __len__(self):
        return len(self.patients) 
    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")    
    def __getitem__(self, index):
        patient = self.patients[index]
        patient_df = self.csv.loc[self.csv.ID == patient].sort_values('order')
        LABEL = patient_df.LABEL.to_list()[0]
        ID = patient_df.ID.to_list()[0]
        jpg_lst = patient_df.file_name.to_list()
        jpg_lst = [os.path.join(args.data_dir, 'external_jpg_cbct_90_de'+str(args.fold), "Ex_" + LABEL + "_" + f'{ID:08}', f"{f:04}.jpg") for f in jpg_lst]
        img_lst = [cv2.imread(jpg)[:,:,::-1] for jpg in jpg_lst] 
        img_lst = [transform2D(image=img)['image'] for img in img_lst] # [T,H,W,C]
        img = np.stack([image.astype(np.float32) for image in img_lst], axis=2).transpose(3,0,1,2) #[C,H,W,T]
        if self.transform is not None:
            img = apply_transform(self.transform, img) #[C,H,W,T]
        img_right = img[:, :, :int(args.resize), :] #[C,H,W,T]
        img_left = img[:, :, int(args.resize):, :] #[C,H,W,T]
        img_left = torch.flip(img_left, [2])
        target_right = patient_df.Rt.to_list()[0]
        target_left = patient_df.Lt.to_list()[0]
        ID_right = ID
        ID_left = ID + 0.1
        if LABEL == "F":
            target_right = 2 if target_right == 1 else 0
            target_left = 2 if target_left == 1 else 0
        elif LABEL == "S":
            target_right = 1 if target_right == 1 else 0
            target_left = 1 if target_left == 1 else 0
        elif LABEL == "N":
            target_right = 0
            target_left = 0
        if self.mode == 'test':
            return img
        else:
            return ID_right, ID_left, img_right, img_left, target_right, target_left
        
def mod_index(df):
    index_list = []
    for idx in df.index:
        int_idx = int(idx)
        if idx % 1 == 0:
            hand = "_right"
        else:
            hand = "_left"
        mod_idx = str(int_idx) + hand
        index_list.append(mod_idx)
    df.index = index_list
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default= 711)
    parser.add_argument('--data_dir', type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/input_external/")
    parser.add_argument('--crop_size', type=int, default=300) # x-axis, y-axis
    parser.add_argument('--resize', type=int, default=160) # x-axis, y-axis
    parser.add_argument('--z_resize', type=int, default=160) # z-axis 
    parser.add_argument('--DEBUG', type=str, default="F")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet18") # densenet121, resnet18, resnet34,resnet50 
    parser.add_argument("--df_dir", type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/3_Codes_external/key_logs/cbct_90_denoising/")
    parser.add_argument("--key_thresh", type=float, default=0.5)
    parser.add_argument("--save_name", type=str, default="cbct_90_denoising_resnet18_batch16_aug0.5_key0.5")

    args, _ = parser.parse_known_args()
    print(args)
    
fold = args.fold

args.weight_dir = glob(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D/3D_weights/",args.save_name,f"*best_auc_fold{fold}.pth"))[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df_dir = os.path.join(args.df_dir,f"df_*_fold{fold}.csv")
df_dir = glob(df_dir)[0]
df = pd.read_csv(df_dir)
print(f"Read {df_dir}")

ID_list = df.ID.unique()
df["key_target"]=-1
df['order'] = df.file_name
for ID in ID_list:
    ID_df = df[(df.ID==ID) & (df.ref_pred > args.key_thresh)].copy()
    KEY_START = ID_df.order.min()
    KEY_END = ID_df.order.max()
    df.loc[(df.ID==ID) & (df.order >= KEY_START) & (df.order <= KEY_END),'key_target'] = 1
df = df[df.key_target == 1].reset_index(drop=True)

df['target']=-1
df['LABEL'] = df.Label
df.loc[df.LABEL=="F","target"] = 2
df.loc[df.LABEL=="S","target"] = 1
df.loc[df.LABEL=="N","target"] = 0

transform2D = albumentations.Compose([albumentations.CenterCrop(height = args.crop_size, width = args.crop_size, p=1)])

val_transforms = Compose([ScaleIntensity(), 
                          Resize((args.resize, args.resize * 2, args.z_resize)),
                          ToTensor()])


if args.DEBUG == "T":
    print('DEBUGING...')
    df = df[(df.ID == df.ID.unique()[0]) | (df.ID == df.ID.unique()[1])| (df.ID == df.ID.unique()[2])].reset_index(drop=True)
    args.n_epochs = 3
    
dataset_external = OMUDataset3D(df, 'val', transform=val_transforms)
loader = torch.utils.data.DataLoader(dataset_external, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
model=get_models()

print(len(dataset_external))
save_dir = os.path.join("./cam", args.save_name, f'fold{fold}')

if os.path.exists(save_dir) == False:
    model = medcam.inject(model, output_dir=save_dir, backend='gcam', layer='layer4', label='best',
                          save_maps=True, return_score=True)
    model.eval()
    PREDS = []
    TARGETS = []
    IDS = []
    for (ID_right, ID_left, img_right, img_left, target_right, target_left) in tqdm(loader):
        pred = model(img_right)
        pred = np.argmax(pred[0].detach().numpy())
        PREDS.append(pred)
        pred = model(img_left)
        pred = np.argmax(pred[0].detach().numpy())
        PREDS.append(pred)
        target = torch.cat((target_right, target_left),0)
        ID = torch.cat((ID_right, ID_left),0)
        TARGETS.append(target)
        IDS.append(ID)
    TARGETS = torch.cat(TARGETS).numpy()
    IDS = torch.cat(IDS).numpy()
    df_pred = pd.DataFrame({"Pred": PREDS, "GT": TARGETS, "Answer": PREDS==TARGETS},index = IDS)
    df_pred = mod_index(df_pred)
    df_pred.to_csv(os.path.join(save_dir, f"df_pred_fold{fold}.csv"))
else:
    df_pred = pd.read_csv(os.path.join(save_dir, f"df_pred_fold{fold}.csv"))
    PREDS = df_pred.Pred
    
val_cam_transforms = Compose([ScaleIntensity(), 
                              Resize((args.resize, args.resize, args.z_resize), mode='trilinear'),
                              ToTensor()])

for idx in tqdm(range(len(loader))):
    for view in ["coronal"]: #, "axial", "sagittal"]:
        os.makedirs(os.path.join(save_dir, f'layer4_{view}'), exist_ok=True)
        gcam_dict = {}
        raw_dict = {}
        label_dict = {}
        for side in range(0,2):
            nii_gz=glob(os.path.join(save_dir, 'layer4', f'attention_map_{2*idx + side}_*'))[0]
            gcam_img = nib.load(nii_gz).get_fdata() # T, W, H
            gcam_img = gcam_img.transpose(2,1,0) # H, W, T
            gcam_img = np.expand_dims(gcam_img, axis=0) # C, H, W, T
            gcam_img = apply_transform(val_cam_transforms, gcam_img)
            if side == 0: gcam_dict["right"] = gcam_img
            else: gcam_dict["left"] = gcam_img
        ID_right, ID_left, right_img, left_img, right_label, left_label = dataset_external[idx]
        raw_dict["right"] = right_img # C, H , W, T
        raw_dict["left"] = left_img # C, H , W, T
        label_dict["right"] = right_label
        label_dict["left"] = left_label
        eval_dict = {}
        pred_dict = {}
        for side in ["right","left"]:
            img_list=[]
            if side == "right": pred_dict[side] = PREDS[2*idx]
            else: pred_dict[side] = PREDS[2*idx + 1]
            eval_dict[side] = pred_dict[side]==label_dict[side]
            if view == "coronal": axis_size = args.z_resize
            else: axis_size = args.resize
            for j in range(axis_size):
                if view == "coronal":
                    raw_plt = raw_dict[side].numpy().transpose(1,2,3,0)[:,:,j] # H, W, T, C
                    gcam_plt = gcam_dict[side].numpy().transpose(1,2,3,0)[:,:,j] # H, W, T, C
                elif view == "axial":
                    raw_plt = raw_dict[side].numpy().transpose(1,2,3,0)[j,:,:] # H, W, T, C
                    gcam_plt = gcam_dict[side].numpy().transpose(1,2,3,0)[j,:,:] # H, W, T, C
                elif view == "sagittal":
                    raw_plt = raw_dict[side].numpy().transpose(1,2,3,0)[:,j,:] # H, W, T, C
                    gcam_plt = gcam_dict[side].numpy().transpose(1,2,3,0)[:,j,:] # H, W, T, C
                cmap = cm.jet_r(gcam_plt[...,0])[..., :3]
                img_plt = (cmap.astype(np.float) + raw_plt.astype(np.float))/2
                img_plt = img_plt * 255
                img_plt = cv2.cvtColor(img_plt.astype('uint8'), cv2.COLOR_BGR2RGB)
                img_list.append(img_plt)

            img_array = np.array(img_list)
            imageio.mimsave(os.path.join(save_dir, f'layer4_{view}', 
                                         f"{ID_right}_{side}_{pred_dict[side]}_{label_dict[side]}_{eval_dict[side]}.gif"), 
                            img_array.astype('uint8'), duration=0.04)
