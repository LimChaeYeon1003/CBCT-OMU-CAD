import argparse
import random
import numpy as np
import torch
import os
import shutil
import pandas as pd
from glob import glob
import albumentations
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import time
from pylab import rcParams
import sys
from pretrainedmodels import se_resnext101_32x4d, se_resnext50_32x4d
from torch.utils.data import DataLoader


class OMUDataset2D(Dataset):
    def __init__(self, csv, labels, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        self.labels = labels
    def __len__(self):
        return self.csv.shape[0]
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(glob(os.path.join(args.data_dir, "train_jpg","success_cbct_90"+"_de"+str(args.fold),row.LABEL+ "_" + f'{row.ID:08}', row.file_name + ".jpg"))[0])
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image']
        if self.labels:
            target = torch.tensor(self.csv.loc[index, 'target']).float()
            return image, target
        else: return image
        
class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        if args.model in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            model = EfficientNet.from_name(f'efficientnet-{args.model}')
            weight_dir = glob(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/1_Codes_pre/checkpoints", f'*-{args.model}-*'))
            model.load_state_dict(torch.load(weight_dir[0]))
        elif args.model == "se_resnext50_32x4d":
            model = se_resnext50_32x4d(num_classes=1000, pretrained=None)
            model.load_state_dict(torch.load(os.path.join("checkpoints","se_resnext50_32x4d-a260b3a4.pth")))
        elif args.model == "se_resnext101_32x4d":
            model = se_resnext101_32x4d(num_classes=1000, pretrained=None)
            model.load_state_dict(torch.load(os.path.join("checkpoints","se_resnext101_32x4d-3b2fe3d8.pth")))
        self.net = model
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if args.model in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            in_features = self.net._fc.in_features
        elif args.model in ["se_resnext50_32x4d", "se_resnext101_32x4d"]:
            in_features = self.net.last_linear.in_features
        self.last_linear = nn.Linear(in_features, 1)
    def forward(self, x):
        if args.model in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            x = self.net.extract_features(x)
            x = self.net._avg_pooling(x)
        elif args.model in ["se_resnext50_32x4d", "se_resnext101_32x4d"]:
            x = self.net.features(x)
            x = self.avg_pool(x)
        
        feature = x.view(x.size(0), -1)
        x = self.last_linear(feature)
        return feature, x
    
def predict_df(model, loader, get_output=False):
    model.eval()
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            if torch.cuda.is_available(): data, target = data.to(device), target.to(device)
            _, logits = model(data)
            LOGITS.append(logits.squeeze().detach().cpu())
            TARGETS.append(target.detach().cpu())
            del data, target, logits
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    return LOGITS, PROBS, TARGETS


def main():
    global device, df

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): scaler = torch.cuda.amp.GradScaler()
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+"key_logs", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+"key_weights", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+f"key_logs/{args.kernel_type}", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+f"key_weights/{args.kernel_type}", exist_ok=True)
    with open(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+f'key_logs/{args.kernel_type}/set_{args.kernel_type}_fold{args.fold}.txt', 'a') as appender:
        appender.write(str(args) + '\n')
    df = pd.read_csv(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/1_Codes_pre/fold_df.csv"))
    jpg_list = glob('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/input/train_jpg/success_cbct_90_"+"de"+str(args.fold)+"/*/*")
    jpg_list = [jpg.replace("\\","/") for jpg in jpg_list]
    ID_list = [int(jpg.split("/")[-2].split("_")[1]) for jpg in jpg_list]
    jpg_list = [jpg.split("/")[-1][:-4] for jpg in jpg_list]
    df_jpg = pd.DataFrame({"ID": ID_list, "file_name": jpg_list})
    df = df.merge(df_jpg, on="ID")
    df['order'] = df.file_name.str[-5:].astype(int)
    df['target'] = 0
    ID_list = df[df.LABEL!="F"].ID.unique()
    for ID in ID_list:
        d =df[df.ID == ID].reset_index().copy()
        start = d.ROI_START[0]
        end = d.ROI_END[0]
        df.loc[(df.ID==ID) & (df.order >= start) & (df.order <= end), 'target'] = 1

    transforms_val = albumentations.Compose([
            # albumentations.CenterCrop(height = args.image_size, width = args.image_size, p=1), # 나중에 주석해제
            albumentations.Resize(300,300),
            albumentations.Normalize(p=1),
            ToTensorV2(),
    ])

    if args.DEBUG == "T":
        print('DEBUGING...')
        if len(df) > args.batch_size*2: df = df.sample(args.batch_size * 2).reset_index(drop=True) 
        args.n_epochs = 3

    dataset_df = OMUDataset2D(df, True, transform=transforms_val)
    df_loader = DataLoader(dataset_df, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_model()

    if args.weight_dir is not None: 
        print(f"Load '{args.weight_dir}'...")
        try:  # single GPU model_file
            model.load_state_dict(torch.load(args.weight_dir), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(args.weight_dir)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
    else:
        sys.exit()


    model = model.to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    _, PROBS, TARGET = predict_df(model, df_loader)
    df['ref_pred']=PROBS
    df = df.drop(["target"], axis=1)
    df = df.sort_values(["ID",'order'],ascending=[True, True])
    df.to_csv(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/"+f"key_logs/{args.kernel_type}/df_{args.kernel_type}_fold{args.fold}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/input/")
    parser.add_argument('--image_size', type=int, default=300) # 512, 250
    parser.add_argument('--kernel_type', type=str, default="TEST")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--weight_dir", type=str, default="/home/mars/workspace/cy_workspace/OMU/Classfication_OMU/1_Codes_pre/key_weights/11_22_DEBUG/11_22_DEBUG_best_fold0.pth") 
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="b0") # b0-b7, se_resnext50_32x4d, se_resnext101_32x4d
    parser.add_argument("--DEBUG", type=str, default="T")
    args, _ = parser.parse_known_args()
    print(args)
    main()