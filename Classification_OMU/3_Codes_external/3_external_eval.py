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
matplotlib.use('Agg')
from matplotlib.ticker import MaxNLocator
from models import resnet
from models.model import (generate_model, load_pretrained_model, make_data_parallel, get_fine_tuning_parameters)
import albumentations
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import roc_auc_score, auc

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
        LABEL = patient_df.Label.to_list()[0]
        ID = patient_df.ID.to_list()[0]
        jpg_lst = patient_df.file_name.to_list()
        jpg_lst = [os.path.join(args.data_dir, 'external_jpg_cbct_90', "Ex_" + f"{LABEL}_{ID:08d}", f"{f:04}.jpg") for f in jpg_lst]
        #print(jpg_lst)
        img_lst = [cv2.imread(jpg) for jpg in jpg_lst] 
        # transform2D = albumentations.Compose([albumentations.Crop(x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,p=1)])
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

def val_epoch(model, loader, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    IDS = []
    with torch.no_grad():
        for (ID_right, ID_left, img_right, img_left, target_right, target_left) in tqdm(loader): #B, C, H, W, T
            data = torch.cat((img_right, img_left),0).to(device) # 2xB, C, H, W, T
            target = torch.cat((target_right, target_left),0)
            ID = torch.cat((ID_right, ID_left),0)
            data, target = data.to(device), target.to(device)
            logits = model(data)
            LOGITS.append(logits.detach().cpu())
            TARGETS.append(target.detach().cpu())
            IDS.append(ID.cpu())
            
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    IDS = torch.cat(IDS).numpy()

    acc = (np.argmax(PROBS, axis=1) == TARGETS).mean() * 100.
    try: micro_auc, macro_auc = compute_auc(TARGETS, PROBS)
    except: micro_auc, macro_auc = 0, 0
    return acc, micro_auc, macro_auc, PROBS, TARGETS, IDS

def compute_auc(TARGETS, PROBS):
    n_classes=3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute Micro AUC
    true_y = convert_onehot(TARGETS)
    fpr["micro"], tpr["micro"], _ = roc_curve(true_y.ravel(), PROBS.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute Macro AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_y[:, i], PROBS[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["micro"], roc_auc["macro"]

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

def run():
    global device, df, val_transforms, transform2D, dataset_external
    print(f"Fold: {fold}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_dir = os.path.join(args.df_dir,f"df_*_fold{fold}.csv")
    df_dir = glob(df_dir)[0] #df_external_0225_SIZE300_0
    print(f"Read {df_dir}")
    df = pd.read_csv(df_dir)
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
    df.loc[df.Label=="F","target"] = 2
    df.loc[df.Label=="S","target"] = 1
    df.loc[df.Label=="N","target"] = 0


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
    model = model.to(device)

    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    # print(len(dataset_train), len(dataset_valid))
    val_loss_best, val_acc_best, val_auc_best = 1000, -0.5, -0.5
    val_acc, micro_auc, macro_auc, PROBS, TARGET, IDS = val_epoch(model, loader)
    return IDS, PROBS, TARGET, val_acc, micro_auc, macro_auc

    
def save_confusion(TARGET, PROB, fold):
    cr = classification_report(TARGET, np.argmax(PROB, axis=1), digits=4)
    cm = np.array2string(confusion_matrix(TARGET, np.argmax(PROB, axis=1)))
    with open(os.path.join(save_dir, f'report_fold{fold}.txt'), "w") as f: 
        f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))

def save_auc(macro_list, micro_list, acc_list, folds):
    for idx, fold in enumerate(folds):
        with open(os.path.join(save_dir, f'macro_auc.txt'), 'a') as appender: 
            appender.write(f'Fold {fold}: ' + str(macro_list[idx]) + '\n')
        with open(os.path.join(save_dir, f'micro_auc.txt'), 'a') as appender:
            appender.write(f'Fold {fold}: ' + str(micro_list[idx])+ '\n')
        with open(os.path.join(save_dir, f'acc.txt'), 'a') as appender: 
            appender.write(f'Fold {fold}: ' + str(acc_list[idx])+ '\n')

    with open(os.path.join(save_dir, f'macro_auc.txt'), 'a') as appender: 
        appender.write(f'Mean: ' + str(np.mean(macro_list)) + '\n')
    with open(os.path.join(save_dir, f'micro_auc.txt'), 'a') as appender: 
        appender.write(f'Mean: ' + str(np.mean(micro_list)) + '\n')
    with open(os.path.join(save_dir, f'acc.txt'), 'a') as appender: 
        appender.write(f'Mean: ' + str(np.mean(acc_list)) + '\n')
        
def save_roc(TARGET_list, PROB_list, folds):
    for way in ['micro','macro']:
        plt.cla()
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        tprs = []
        aucs = []
        for idx, fold in enumerate(folds):
            fpr, tpr = get_fpr_tpr(TARGET_list[idx], PROB_list[idx])
            tprs.append(np.interp(mean_fpr, fpr[way], tpr[way]))
            mean_tpr += np.interp(mean_fpr, fpr[way], tpr[way])
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr[way], tpr[way])
            aucs.append(roc_auc)
            plt.plot(fpr[way], tpr[way], lw=1, label='ROC fold %d (area = %0.3f)' % (fold, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        mean_tpr /= len(folds)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        plt.plot(mean_fpr, mean_tpr, 'k--', lw=2, label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, np.std(aucs)))
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='mediumturquoise', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, f'external_{way}_roc_fold{args.fold}.jpg'))
        
def get_fpr_tpr(TARGETS, PROBS):
    n_classes=3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute Micro AUC
    true_y = convert_onehot(TARGETS)
    fpr["micro"], tpr["micro"], _ = roc_curve(true_y.ravel(), PROBS.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute Macro AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_y[:, i], PROBS[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    return fpr, tpr


    
def save_pred_df(PROB, TARGET, IDS):
    df_probs = pd.DataFrame(PROB, index=IDS, columns=["Normal_prob","Non-fungal_prob","Fungal_prob"])
    df_probs['prob'] = np.argmax(PROB, axis=1)
    df_targets = pd.DataFrame(convert_onehot(TARGET), index=IDS, columns=["Normal_gt","Non-fungal_gt","Fungal_gt"])
    df_targets['target'] = TARGET
    df = pd.merge(df_probs, df_targets, left_index=True, right_index=True)
    df = df.sort_index()
    df['answer'] = df.target == df.prob
    df = mod_index(df)
    df.to_csv(os.path.join(save_dir, f'probs_fold{fold}.csv'))
    
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

def main():
    global fold, df, save_dir
    folds = [0,1,2,3,4] if args.fold==5 else [args.fold]
    os.makedirs('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/3_Codes_external/eval", exist_ok=True)
    save_dir = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/3_Codes_external/eval", args.save_name)
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'set_args.txt'), 'w') as appender:
        appender.write(str(args) + '\n')

    macro_list=[]
    micro_list=[]
    acc_list=[]
    TARGET_list=[]
    PROB_list=[]

    for fold in folds:
        args.weight_dir = glob(os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/2_Codes_3D/3D_weights", args.save_name,f"*best_auc_fold{fold}.pth"))[0]
        IDS, PROB, TARGET, val_acc, micro_auc, macro_auc = run()
        macro_list.append(macro_auc)
        micro_list.append(micro_auc)
        acc_list.append(val_acc)
        TARGET_list.append(TARGET)
        PROB_list.append(PROB)
        save_pred_df(PROB, TARGET, IDS)
        save_confusion(TARGET, PROB, fold)

    save_auc(macro_list, micro_list, acc_list, folds)
    save_roc(TARGET_list, PROB_list, folds)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default= 711)
    parser.add_argument('--data_dir', type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+'/input_external/')
    parser.add_argument('--crop_size', type=int, default=300) # x-axis, y-axis
    parser.add_argument('--resize', type=int, default=160) # x-axis, y-axis
    parser.add_argument('--z_resize', type=int, default=160) # z-axis 
    parser.add_argument('--DEBUG', type=str, default="F")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet18") # densenet121, resnet18, resnet34,resnet50 
    parser.add_argument("--df_dir", type=str, default='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])+"/3_Codes_external/key_logs/cbct_90_noise/")
    parser.add_argument("--key_thresh", type=float, default=0.5)
    parser.add_argument("--save_name", type=str, default="cbct_90_noise_resnet18_batch16_aug0.5_key0.5")

    args, _ = parser.parse_known_args()
    print(args)
    main()
