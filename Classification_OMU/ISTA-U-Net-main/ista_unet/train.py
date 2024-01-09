import os, sys, uuid, torch
from os import path
import numpy as np
import random
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import pickle 
from torch.utils.data import Dataset as TorchDataset

from models import ista_unet
from load_dival_datasets import RandomAccessTorchDataset, get_dataloaders_ct
from utils1 import seed_everything
from dival.measure import PSNR
from ISTA import dataset_dir
from ISTA import model_save_dir

parser = argparse.ArgumentParser(description='ISTA_unet training')
loaders_dataset = RandomAccessTorchDataset()
loaders = DataLoader(loaders_dataset,batch_size=1,num_workers=0,shuffle=True)
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)


    print('==> Making model..')


    model_setting_dict = {'kernel_size' : args.KERNEL_SIZE,
                          'hidden_layer_width_list': args.HIDDEN_WIDTHS,
                          'n_classes': args.IMAGE_CHANNEL_NUM, 
                          'ista_num_steps': args.ISTA_NUM_STEPS,  
                          'lasso_lambda_scalar': args.LASSO_LAMBDA_SCALAR, 
                          'uncouple_adjoint_bool': args.UNCOUPLE_ADJOINT_BOOL,
                          'relu_out_bool': args.RELU_OUT_BOOL}

    model = ista_unet( ** model_setting_dict)    
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.BATCH_SIZE = int(args.BATCH_SIZE / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Preparing data..')
    dataset_setting_dict = {'batch_size': args.BATCH_SIZE, 
                        'num_workers': args.num_workers,
                        'distributed_bool': True}
    loaders_dataset = RandomAccessTorchDataset()
    loaders = DataLoader(loaders_dataset,batch_size=1,num_workers=0,shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max= args.NUM_EPOCH,
                eta_min= 1e-5)


    fit_setting_dict = {'num_epochs': args.NUM_EPOCH,
                    'criterion': eval(args.LOSS_STR), 
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'device': args.gpu} 
def fit_model_with_loaders(model, optimizer, scheduler, num_epochs, criterion, loaders, device, max_grad_norm = 1, seed = 0):

    train_loader = loaders['train'] 
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)
    
    print('start training')
    for epoch in tqdm(range(num_epochs) ) :
        loss = 0
        for i, (x, d) in enumerate(train_loader):
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d) 
            
            # compute accumulated gradients
            train_loss.backward()
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 

        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 

    return model



def fit_model_with_loaders_verbose_iter(model, optimizer, scheduler, num_epochs, criterion, loaders, device, verbose_every = 100, max_grad_norm = 1, seed = 0):

    train_loader = loaders['train']    
    len_train_loader = len(train_loader) 
    seed_everything(seed = seed)
    
    print('start training')
    for epoch in tqdm(range(num_epochs) ) :
        loss = 0
        for i, (x, d) in enumerate(train_loader):
            x, d = x.cuda(device), d.cuda(device)
                    
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(x)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, d)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)            

            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss +=  float(train_loss) 
            
            if i % verbose_every == 0:
                print("iter : {}/{}, loss = {:.6f}".format(epoch * len(loaders['train']) + i, len(loaders['train']) * num_epochs, float(train_loss)))
         
        # compute the epoch training loss
        loss = float(loss) / len_train_loader   
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        
        # update the step-size
        scheduler.step() 

    return model
