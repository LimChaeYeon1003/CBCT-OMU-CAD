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
        image = cv2.imread(glob(os.path.join(args.data_dir, "external_jpg_cbct_90_de"+str(args.fold), "Ex_" + row.Label+ "_" + f'{row.ID:08}', row.file_name + ".jpg"))[0])
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
    with torch.no_grad():
        for data in tqdm(loader):
            if torch.cuda.is_available(): data = data.to(device)
            _, logits = model(data)
            LOGITS.append(logits.squeeze().detach().cpu())
            
            del data, logits
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    return LOGITS, PROBS
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/input_external/')
    parser.add_argument('--image_size', type=int, default=300) # 300
    parser.add_argument('--kernel_type', type=str, default="03_31_cbct_90")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="b0") # b0-b7, se_resnext50_32x4d, se_resnext101_32x4d
    args, _ = parser.parse_known_args()
    print(args)
    
for args.fold in [0,1,2,3,4]:
    args.weight_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+f"/1_Codes_pre/key_weights/{args.kernel_type}/{args.kernel_type}_best_fold{args.fold}.pth"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): scaler = torch.cuda.amp.GradScaler()

    os.makedirs('/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[0:-1])+"/Classification_OMU/3_Codes_external/key_logs", exist_ok=True)
    os.makedirs('/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[0:-1])+f"/Classification_OMU/3_Codes_external/key_logs/{args.kernel_type}", exist_ok=True)
    with open('/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[0:-1])+f'/Classification_OMU/3_Codes_external/key_logs/{args.kernel_type}/set.txt', 'a') as appender:
        appender.write(str(args) + '\n')


    df = pd.read_csv(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/input_external/outputs/external_newID.csv'))

    jpg_list = glob('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/input_external/external_jpg_cbct_90_de"+str(args.fold)+"/*/*")
    jpg_list = [jpg.replace("\\","/") for jpg in jpg_list]
    ID_list = [int(jpg.split("/")[-2].split("_")[2]) for jpg in jpg_list]
    jpg_list = [jpg.split("/")[-1][:-4] for jpg in jpg_list]
    df_jpg = pd.DataFrame({"ID": ID_list, "file_name": jpg_list})
    df = df.merge(df_jpg, on="ID")

    transforms_val = albumentations.Compose([
            albumentations.CenterCrop(height = args.image_size, width = args.image_size, p=1),
            albumentations.Normalize(p=1),
            ToTensorV2(),
    ])

    dataset_df = OMUDataset2D(df, False, transform=transforms_val)
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
    _, PROBS = predict_df(model, df_loader)
    df['ref_pred']=PROBS
    df.to_csv('/'.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('/')[0:-1])+f'/Classification_OMU/3_Codes_external/'+f"/key_logs/{args.kernel_type}/df_external_{args.kernel_type}_fold{args.fold}.csv", index=False)