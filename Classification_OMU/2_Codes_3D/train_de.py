import random
import torch
import argparse
import numpy as np
import pandas as pd
import os
import monai
# from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor
from monai.transforms import Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine, CenterSpatialCrop, RandFlip
from monai.utils import get_seed
from glob import glob
import cv2
from pylab import rcParams
import torch.nn as nn
import torch.optim as optim
from warmup_scheduler.scheduler import GradualWarmupScheduler
from tqdm import tqdm as tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import time
import csv
import shutil
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.ticker import MaxNLocator
from models import resnet
from models.model import (generate_model, load_pretrained_model, make_data_parallel, get_fine_tuning_parameters)
import albumentations
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import nibabel as nib
warnings.simplefilter("ignore", UserWarning)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
        
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        
def convert_onehot(target):
    GTS=[]
    for GT in target:
        if GT == 0:
            GTS.append([1, 0, 0])
        elif GT ==1:
            GTS.append([0, 1, 0])
        elif GT == 2:
            GTS.append([0, 0, 1])
    GTS = torch.LongTensor(GTS)
    return GTS.numpy()

def log_csv(epoch, train_loss, val_loss, val_acc, val_auc):
    result = [epoch, train_loss, val_loss, val_acc, val_auc]
    with open(csv_file, 'a', newline='') as f:        
        writer = csv.writer(f)
        if epoch == 1: writer.writerow(["Epochs", "trn_loss", "val_loss", "val_acc", "val_auc"])
        writer.writerow(result)   

def get_models():
    if args.model == "densenet121": 
        model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=3, out_channels=3)
    elif args.model == "resnet18":  
        model = resnet.generate_model(model_depth=18, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/2_Codes_3D/models/r3d18_K_200ep.pth', 'resnet', 3)
    elif args.model == "resnet34":  
        model = resnet.generate_model(model_depth=34, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/2_Codes_3D/models/r3d34_K_200ep.pth', 'resnet', 3)
    elif args.model == "resnet50":  
        model = resnet.generate_model(model_depth=50, n_classes=700, n_input_channels=3)
        model = load_pretrained_model(model, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/2_Codes_3D/models/r3d50_K_200ep.pth', 'resnet', 3)
    if args.weight_dir is not None: 
        print(f"Load '{args.weight_dir}'...")
        try:  # single GPU model_file
            model.load_state_dict(torch.load(args.weight_dir), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(args.weight_dir)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
    return model
        
def plt_show():
    dataset_show = OMUDataset3D(df, 'train', transform=val_transforms)
    dataset_show_aug = OMUDataset3D(df, 'train', transform=train_transforms)
    rcParams['figure.figsize'] = 20,5
    for i in range(5):
        f, axarr = plt.subplots(1,6)
        img_right, img_left, target_right, target_left = dataset_show[i]
        for j in range(6):        
            if j==0: axarr[j].imshow(img_right.numpy().transpose(1,2,3,0)[args.resize//3,:,:]) #[C,H,W,T]
            elif j==1: axarr[j].imshow(img_right.numpy().transpose(1,2,3,0)[:,args.resize//3,:]) #[C,H,W,T]
            elif j==2: axarr[j].imshow(img_right.numpy().transpose(1,2,3,0)[:,:,args.z_resize//3]) #[C,H,W,T]
            elif j==3: axarr[j].imshow(img_left.numpy().transpose(1,2,3,0)[args.resize//3,:,:])
            elif j==4: axarr[j].imshow(img_left.numpy().transpose(1,2,3,0)[:,args.resize//3,:])
            elif j==5: axarr[j].imshow(img_left.numpy().transpose(1,2,3,0)[:,:,args.z_resize//3])
            axarr[j].set_title(f"Orig {i}")
            axarr[j].axis('off')
        
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
        jpg_lst = [os.path.join(args.data_dir, 'train_jpg', "success_cbct_90"+"_de"+str(args.fold),  f"{LABEL}_{ID:08d}", f"{f}.jpg") for f in jpg_lst] 
        img_lst = [cv2.imread(jpg)[:,:,::-1] for jpg in jpg_lst]
        img_lst = [transform2D(image=img)['image'] for img in img_lst] # [T,H,W,C]
        img = np.stack([image.astype(np.float32) for image in img_lst], axis=2).transpose(3,0,1,2) #[C,H,W,T]
        if self.transform is not None:
            img = apply_transform(self.transform, img) #[C,H,W,T]
        img_right = img[:, :, :int(args.resize), :] #[C,H,W,T]
        img_left = img[:, :, int(args.resize):, :] #[C,H,W,T]
        img_left = torch.flip(img_left, [2])
        img = [img_right, img_left]
        target_right = patient_df.Rt.to_list()[0]
        target_left = patient_df.Lt.to_list()[0]
        if LABEL == "F":
            target_right = 2 if target_right == 1 else 0
            target_left = 2 if target_left == 1 else 0
        elif LABEL == "S":
            target_right = 1
            target_left = 1
        elif LABEL == "N":
            target_right = 0
            target_left = 0
        target = [torch.tensor(target_right).long(), torch.tensor(target_left).long]
        if self.mode == 'test':
            return img
        else:
            TARGET = patient_df['target'].to_list()[0]
            return img_right, img_left, target_right, target_left
        
def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (img_right, img_left, target_right, target_left) in bar: #B, C, H, W, T
        data = torch.cat((img_right, img_left),0).to(device) # 2xB, C, H, W, T
        target = torch.cat([target_right, target_left]).to(device)
        optimizer.zero_grad()
        logits = model(data)       
        loss = criterion(logits, target)
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
        for (img_right, img_left, target_right, target_left) in loader: #B, C, H, W, T
            data = torch.cat((img_right, img_left),0).to(device) # 2xB, C, H, W, T
            target = torch.cat([target_right, target_left]).to(device)
            data, target = data.to(device), target.to(device)
            logits = model(data)
            LOGITS.append(logits.detach().cpu())
            TARGETS.append(target.detach().cpu())
            del data, target, logits
    val_loss = criterion(torch.cat(LOGITS), torch.cat(TARGETS)).numpy()
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    acc = (np.argmax(PROBS, axis=1) == TARGETS).mean() * 100.
    try: auc = roc_auc_score(convert_onehot(TARGETS).ravel(), PROBS.ravel())
    except: auc = 0
    return float(val_loss), acc, auc, PROBS, TARGETS

def run():
    global device, criterion, scaler, df, val_transforms, train_transforms, transform2D,dataset_valid
    print(f"Fold: {fold}")
    if args.SEED is not None: set_seed(args.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): scaler = torch.cuda.amp.GradScaler()
    os.makedirs('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+"/3D_logs", exist_ok=True)
    os.makedirs('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+"/3D_weights", exist_ok=True)
    os.makedirs('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f"/3D_logs/{args.save_name}", exist_ok=True)
    os.makedirs('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f"/3D_weights/{args.save_name}", exist_ok=True)
    with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_logs/{args.save_name}/set_{args.save_name}_fold{args.fold}.txt', 'a') as appender:
        appender.write(str(args) + '\n')
    df_dir = args.df_dir
    df_dir = glob(os.path.join(df_dir,f"df_*_fold{fold}.csv"))[0]
    df = pd.read_csv(df_dir)
    ID_list = df.ID.unique()
    df["key_target"]=-1
    for ID in ID_list:
        ID_df = df[(df.ID==ID) & (df.ref_pred > args.key_thresh)].copy()
        KEY_START = ID_df.order.min()
        KEY_END = ID_df.order.max()
        df.loc[(df.ID==ID) & (df.order >= KEY_START) & (df.order <= KEY_END),'key_target'] = 1
    df = df[df.key_target == 1].reset_index(drop=True)

    df['target']=-1
    df.loc[df.LABEL=="F","target"] = 2
    df.loc[df.LABEL=="S","target"] = 1
    df.loc[df.LABEL=="N","target"] = 0

    transform2D = albumentations.Compose([albumentations.CenterCrop(height = args.crop_size, width = args.crop_size, p=1)])
    train_transforms = Compose([ScaleIntensity(), 
                                    Resize((args.resize, args.resize*2 ,args.z_resize)), 
                                    RandAffine(prob=args.augment,
                                                translate_range=(5, 5, 5),
                                                rotate_range=(0, 0, 0),
                                                scale_range=(0.1, 0.1, 0.1),
                                                padding_mode='border'),
                                    ToTensor()])
    val_transforms = Compose([ScaleIntensity(), 
                              Resize((args.resize, args.resize * 2, args.z_resize)),
                              ToTensor()])

    if args.plt_show == "T": 
        plt_show()
        sys.exit()
    criterion = nn.CrossEntropyLoss()

    df_train = df[(df['fold'] != fold)].reset_index(drop=True) 
    df_valid = df[(df['fold'] == fold)].reset_index(drop=True)
    if args.DEBUG == "T":
        print('DEBUGING...')
        df_train = df_train[(df_train.ID == df_train.ID.unique()[0]) | (df_train.ID == df_train.ID.unique()[1])| (df_train.ID == df_train.ID.unique()[2])].reset_index(drop=True)
        df_valid = df_valid[(df_valid.ID == df_valid.ID.unique()[0]) | (df_valid.ID == df_valid.ID.unique()[1])| (df_valid.ID == df_valid.ID.unique()[2])].reset_index(drop=True)
        args.n_epochs = 1

    dataset_train = OMUDataset3D(df_train, 'train', transform=train_transforms)
    dataset_valid = OMUDataset3D(df_valid, 'val', transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model=get_models()
    model = model.to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    cosine_epo = 2 if args.DEBUG == "T" else args.n_epochs - 1
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    print(len(dataset_train), len(dataset_valid))
    val_loss_best, val_acc_best, val_auc_best = 1000, -0.5, -0.5

    for epoch in range(1, args.n_epochs+1):
        scheduler_warmup.step(epoch-1)
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_auc, PROBS, TARGET = val_epoch(model, valid_loader)
        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {(val_loss):.4f}, acc: {(val_acc):.1f}, auc: {(val_auc):.3f}'
        print(content)
        with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_logs/{args.save_name}/log_{args.save_name}_fold{fold}.txt', 'a') as appender: appender.write(content + '\n')             
        log_csv(epoch, train_loss, val_loss, val_acc, val_auc)

        if val_loss < val_loss_best:
            torch.save(model.state_dict(), '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_weights/{args.save_name}/{args.save_name}_best_loss_fold{fold}.pth')
            val_loss_best = val_loss
        if val_acc > val_acc_best:
            torch.save(model.state_dict(), '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_weights/{args.save_name}/{args.save_name}_best_acc_fold{fold}.pth')
            val_acc_best = val_acc
        if val_auc > val_auc_best:
            print('val_auc_best ({:.4f} --> {:.4f}).  Saving model ...'.format(val_auc_best, val_auc))
            torch.save(model.state_dict(), '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_weights/{args.save_name}/{args.save_name}_best_auc_fold{fold}.pth')
            val_auc_best = val_auc
            BEST_PROBS = PROBS
            val_acc_output = val_acc

    return BEST_PROBS, TARGET, val_acc_output

def loss_plot(fold):
    fig, ax = plt.subplots()
    colors = ["deeppink", "gold", "darkolivegreen", "mediumaquamarine", "mediumpurple"]
    
    loss = pd.read_csv(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+"/3D_logs", args.save_name, f'log_{args.save_name}_fold{fold}.csv'))
    loss = loss["val_loss"].tolist()
    ax.plot(range(1,len(loss)+1), loss, lw=1.5, alpha=0.3, color = "blue", label = 'Fold 0{}'.format(fold))
    ax.set(xlim=[1-0.05,len(loss)+0.05], ylim=[0,np.max(loss)+0.1])
    ax.set(xlabel='Epoch', ylabel="val_loss", title="{} vs. epoch".format("val_loss"))
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.figure.savefig('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_logs/{args.save_name}/val_loss_{args.save_name}_fold{fold}.jpg')

def main():
    global fold, csv_file
    fold = args.fold
    csv_file = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+"/3D_logs", args.save_name, f'log_{args.save_name}_fold{fold}.csv')
    if os.path.exists(csv_file): os.remove(csv_file)
    BEST_PROBS, TARGET, val_acc = run()
    content = f'Acc: {(val_acc):.2f}'
    with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_logs/{args.save_name}/val_acc_{args.save_name}_fold{args.fold}.txt', 'a') as appender: appender.write(content + '\n')  

    cr = classification_report(TARGET, np.argmax(BEST_PROBS, axis=1), digits=4)
    cm = np.array2string(confusion_matrix(TARGET, np.argmax(BEST_PROBS, axis=1)))
    with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D"+f'/3D_logs/{args.save_name}/report_{args.save_name}_fold{fold}.txt', "w") as f: 
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

    loss_plot(args.fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default= 711)
    parser.add_argument('--data_dir', type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/input/')
    parser.add_argument('--crop_size', type=int, default=300) # x-axis, y-axis
    parser.add_argument('--resize', type=int, default=160) # x-axis, y-axis
    parser.add_argument('--z_resize', type=int, default=160) # z-axis 
    parser.add_argument('--save_name', type=str, default="03_new_deno")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--plt_show', type=str, default="F")
    parser.add_argument('--DEBUG', type=str, default="F")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_dir", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet18") # densenet121, resnet18, resnet34,resnet50 
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--augment", type=float, default=0.5)
    parser.add_argument("--df_dir", type=str, default=('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1]))+"/1_Codes_pre/key_logs/cbct_90_denoising/")
    parser.add_argument("--key_thresh", type=float, default=0.5)

    args, _ = parser.parse_known_args()
    print(args)
    main()
