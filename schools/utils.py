import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time

# from models import *
from adp_r18 import *
from spaware_models import *
from dataloader import PlanetData

import joblib

from sklearn.metrics import r2_score


def resnet18(num_classes = 1000, normalize = False, use_means = False, use_lc = False):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, normalize = normalize, use_means = use_means, use_lc = use_lc)



def apply_transforms(stats):
    """Define transformations for training and validation data."""
    transforms_dict = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ])
    }
    return transforms_dict


def construct_model(model_name, norm_coords = False, use_means = False, use_lc = False):
    if model_name == "DeepAll":
        model = models.resnet18(pretrained = True)
        model.fc = torch.nn.Linear(512, 1)
    elif model_name == "SpA":
        model = FTSelector()
    elif model_name == "GeoConv":
        model = resnet18(num_classes=1, normalize = norm_coords, use_means = use_means, use_lc = use_lc)
    elif model_name == "FC":
        model = MainModel(512, 512, 1)
    return model


def save_model(folder_name, epoch, state_dict, criterion, optimizer, scheduler, best = False):
    
    if best:
        fname = f"{folder_name}/model_epoch{epoch}.torch"
    else:
        fname = f"{folder_name}/most_recent_epoch{epoch}.torch"  
        
    # Save the most current epoch
    torch.save({
                'epoch': epoch,
                'loss': criterion,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, fname)   




import torch
import torch.nn as nn

class WeightedL1Loss(nn.Module):
    def __init__(self, weight_zero = 0.5, weight_non_zero = 1.5):
        super(WeightedL1Loss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_non_zero = weight_non_zero

    def forward(self, inputs, targets):
        # Compute the absolute differences
        diff = torch.abs(inputs - targets)
        
        # Create a mask for zero and non-zero targets
        mask_zero = (targets == 0).float()
        mask_non_zero = (targets != 0).float()
        
        # Apply weights
        loss_zero = self.weight_zero * diff * mask_zero
        loss_non_zero = self.weight_non_zero * diff * mask_non_zero
        
        # Combine the losses and take the mean
        loss = (loss_zero + loss_non_zero).mean()
        
        return loss

# Example usage:
# criterion = WeightedL1Loss(weight_zero=0.5, weight_non_zero=1.5)
# loss = criterion(predictions, targets)




def get_psuedo_labels(dataloaders, model, device, model_name):

    preds = {}
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        
        data = dataloaders[phase]
        data.dataset.set_stage('test')

        model.eval()   # Set model to evaluate mode

        # Iterate over data.
        for c, (inputs, labels, coords, imname, true_label) in enumerate(data):
        
            inputs = inputs.to(device)
            coords = coords.to(device)
            labels = labels.to(device).view(-1, 1)

            if model_name in ["SpA", "GeoConv", "FC"]:
                outputs = model(inputs, coords)
            else:
                outputs = model(inputs)

            print(c, len(data), device, end = "\r")

            preds[imname[0]] = outputs.item()

    return preds
    
    
        
def bleh(new_preds, labels, coords, epoch, region):

    model = joblib.load("loss_predictor.joblib")
    print(type(model))    

    df = pd.json_normalize(new_preds).T.reset_index()
    df.columns = ["name", "pred"]

    df["label"] = df["name"].map(labels)
    df["coords"] = df["name"].map(coords)
    df["lon"] = df["coords"].str[0]
    df["lat"] = df["coords"].str[1]
    df["DHSID"] = df["name"].str.split("/").str[10].str.split("_").str[0]

    # print(df)

    # dsfa

    glab = df.groupby(["DHSID"])["label"].agg(["mean", "var"]).reset_index()
    g = pd.DataFrame(df.groupby(["DHSID"])["pred"].agg(["mean", "median", "var", "std"])).reset_index()
    g = pd.merge(g, glab, on = "DHSID")
    g["iso"] = g["DHSID"].str[0:2]
    g.columns = ["DHSID", "avg_pred_y", "med_pred_y", "var_pred_y", "std_pred_y", "mean_true_y", "var_true_y", "iso"]
    g["avg_error"] = abs(g["mean_true_y"] - g["avg_pred_y"])
    g["med_error"] = abs(g["mean_true_y"] - g["med_pred_y"])
    g = g.dropna()

    temp_labs = dict(zip(g["DHSID"], g["avg_pred_y"]))
    df["temp_lab"] = df["DHSID"].map(temp_labs)

    df["iso"] = df["name"].str.split("/").str[10].str[0:2]
    df["true_loss"] = df["label"] - df["pred"]

    df['group_std_dev'] = df.groupby('DHSID')['pred'].transform('std')
    df['individual_deviation'] = df['pred'] - df['temp_lab']
    df['group_mean'] = df.groupby('DHSID')['pred'].transform('mean')

    df = df.dropna()

    # Features for predicting loss
    group_std_dev = df['group_std_dev'].values
    individual_deviation = df['individual_deviation'].values
    group_mean = df['group_mean'].values

    # Target: loss
    optimal_epsilons = df["true_loss"].values

    X = np.vstack([group_std_dev, individual_deviation, group_mean]).T
    y = optimal_epsilons

    df["pred_loss"] = model.predict(X)
    df["adjusted_pred"] = df["pred"] + df["pred_loss"]
    df["absval_pred_loss"] = abs(df["pred_loss"])

    print("True r2: ", r2_score(df["label"], df["adjusted_pred"]))

    psuedo = df.sort_values(by = "absval_pred_loss")
    psudeo_labs = dict(zip(psuedo["name"], psuedo["adjusted_pred"]))

    # print("New Labels!")
    # print(psudeo_labs)

    return psudeo_labs, r2_score(df["label"], df["adjusted_pred"])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
