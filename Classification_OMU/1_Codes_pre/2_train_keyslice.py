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
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pylab import rcParams
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score
import sys
from pretrainedmodels import se_resnext101_32x4d, se_resnext50_32x4d
print(os.getcwd())
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True

class OMUDataset2D(Dataset):
    def __init__(self, csv, labels, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        self.labels = labels
    def __len__(self):
        return self.csv.shape[0]
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(glob(os.path.join(args.data_dir, "train_jpg","success_cbct_90",row.LABEL+ "_" + f'{row.ID:08}', row.file_name + ".jpg"))[0])
#         if args.img_show=="T": image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image']
        if self.labels:
#             target = torch.tensor(row.target).float()
            target = torch.tensor(self.csv.loc[index, 'target']).float()
            return image, target
        else: return image

def plt_show():        
    os.makedirs("transform_img",exist_ok=True)
    dataset_show = OMUDataset2D(df, False, transform=transforms_val)
    # dataset_show_aug = OMUDataset2D(df, False, transform=transforms_train)
    rcParams['figure.figsize'] = 6,6
    for i in range(len(df)):
        f, axarr = plt.subplots(1,1)
        img = dataset_show[i]
        axarr.imshow(img.numpy().transpose(1,2,0))
    #     img = dataset_show_aug[i]
    #     axarr[1].imshow(img.numpy().transpose(1,2,0))
        f.savefig(f'transform_img/{df.loc[i,"ID"]}_{df.loc[i,"order"]}.png')

def train_epoch(model, loader, optimizer):
    global logits
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        if torch.cuda.is_available(): data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, logits = model(data)       
        loss = criterion(logits.squeeze(), target)
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        del data, target, logits, loss
    return np.mean(train_loss)

def val_epoch(model, loader, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            if torch.cuda.is_available(): data, target = data.to(device), target.to(device)
            _, logits = model(data)
            LOGITS.append(logits.squeeze().detach().cpu())
            TARGETS.append(target.detach().cpu())
            del data, target, logits
    val_loss = criterion(torch.cat(LOGITS), torch.cat(TARGETS)).numpy()
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    if get_output:
        return LOGITS, PROBS, TARGETS
    else:
        acc = (PROBS.round() == TARGETS).mean() * 100.
        try: auc = roc_auc_score(TARGETS, LOGITS)
        except: auc = 0
        return float(val_loss), acc, auc
    
def log_csv(epoch, train_loss, val_loss, val_acc, val_auc):
    result = [epoch, train_loss, val_loss, val_acc, val_auc]
    with open(os.path.join(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_logs', args.kernel_type, f'log_{args.kernel_type}_fold{args.fold}.csv'), 'a', newline='') as f:        
        writer = csv.writer(f)
        if epoch == 1: writer.writerow(["Epochs", "trn_loss", "val_loss", "val_acc", "val_auc"])
        writer.writerow(result)  
        
def generate_plot():
    for variable in ["val_loss", "val_acc", "val_auc"]:
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        history_file = pd.read_csv(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+f'/1_Codes_pre/key_logs', args.kernel_type, f'log_{args.kernel_type}_fold{args.fold}.csv'))
        history_file["val_acc"] =  history_file["val_acc"]
        history_file = history_file[variable]
        ax.plot(range(1,len(history_file)+1), history_file, lw=1.5, alpha=0.8, color = "blue", label = variable)
        if variable != "val_loss": ax.set(xlim=[1-0.05,len(history_file)+0.05], ylim=[-0.05, 1.05], xlabel='Epoch', ylabel=variable, title=f"{variable} vs. epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.figure.savefig(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_logs/{args.kernel_type}/{variable}_{args.kernel_type}_fold{args.fold}.jpg')

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
    global device, criterion, scaler, df, transforms_train, transforms_val
    if args.SEED is not None: set_seed(args.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): scaler = torch.cuda.amp.GradScaler()
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/key_logs", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/key_weights", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f"/1_Codes_pre/key_logs/{args.kernel_type}", exist_ok=True)
    os.makedirs(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f"/1_Codes_pre/key_weights/{args.kernel_type}", exist_ok=True)
    with open(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_logs/{args.kernel_type}/set_{args.kernel_type}_fold{args.fold}.txt', 'w') as appender:
        appender.write(str(args) + '\n')
    df = pd.read_csv('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/1_Codes_pre/fold_df.csv')

    jpg_list = glob('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/input/train_jpg/success_cbct_90/*/*")
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

#     df_pred = df.copy()
    df = df[df.LABEL!="F"].reset_index(drop=True)


    # crop_p = 0 if args.image_size == 512 else 1
    norm_p = 0 if args.plt_show == "T" else 1

    transforms_train = albumentations.Compose([
            albumentations.Resize(300,300),
            albumentations.Normalize(p=norm_p),
            ToTensorV2(),
    ])
    transforms_val = albumentations.Compose([
            albumentations.Resize(300,300),
            albumentations.Normalize(p=norm_p),
            ToTensorV2(),
    ])

    if args.plt_show == 'T':
        plt_show()
        sys.exit()

    criterion = torch.nn.BCEWithLogitsLoss()
    df_train = df[(df['fold'] != args.fold)].reset_index(drop=True) 
    df_valid = df[(df['fold'] == args.fold)].reset_index(drop=True) 
    if args.DEBUG == "T":
        print('DEBUGING...')
        if len(df_train) > args.batch_size*2: df_train = df_train.sample(args.batch_size * 2).reset_index(drop=True) 
        if len(df_valid) > args.batch_size*2: df_valid = df_valid.sample(args.batch_size * 2).reset_index(drop=True)
        args.n_epochs = 3

    dataset_train = OMUDataset2D(df_train, True, transform=transforms_train)
    dataset_valid = OMUDataset2D(df_valid, True, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model()

    if args.weight_dir is not None: 
        print(f"Load '{args.weight_dir}'...")
        try:  # single GPU model_file
            model.load_state_dict(torch.load(args.weight_dir), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(args.weight_dir)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs)
    print(len(dataset_train), len(dataset_valid))
    model_file = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_weights/{args.kernel_type}/{args.kernel_type}_best_fold{args.fold}.pth'
    val_loss_best=1000
    for epoch in range(1, args.n_epochs+1):
        scheduler_cosine.step(epoch-1)
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_auc = val_epoch(model, valid_loader)
        content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(val_acc):.4f}, auc: {(val_auc):.6f}'
        print(content)
        with open(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_logs/{args.kernel_type}/log_{args.kernel_type}_fold{args.fold}.txt', 'a') as appender: appender.write(content + '\n')
        log_csv(epoch, train_loss, val_loss, val_acc, val_auc)
        if val_loss < val_loss_best:
            print('val_loss_best ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_best, val_loss))
            torch.save(model.state_dict(), model_file)
            val_loss_best = val_loss
        torch.save(model.state_dict(), os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+f'/1_Codes_pre/key_weights/{args.kernel_type}/{args.kernel_type}_epoch{epoch}_fold{args.fold}.pth')

    generate_plot()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default= 711)
    parser.add_argument('--data_dir', type=str, default = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/input/'))
    parser.add_argument('--image_size', type=int, default=300) # 512, 250
    parser.add_argument('--kernel_type', type=str, default="TEST")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--plt_show', type=str, default="F")
    parser.add_argument('--DEBUG', type=str, default="F")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_dir", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--model", type=str, default="b0") # b0-b7, se_resnext50_32x4d, se_resnext101_32x4d
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=30)
    args, _ = parser.parse_known_args()
    print(args)
    main()