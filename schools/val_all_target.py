 # Assuming the ResNet and BasicBlock classes are defined as provided in the previous messages
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
from sklearn.metrics import r2_score


from utils import *
from dataloader import PlanetData


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('region', type = str, help = "can be one of: SpA, DeepAll, GeoConv, FC")
    parser.add_argument('--model_name', required = False, type = str, default = "GeoConv", help = "can be one of: SpA, DeepAll, GeoConv, FC")
    parser.add_argument('--version', required = False, default = 1, type = str)
    parser.add_argument('--eval', action='store_true', help='')
    # parser.add_argument('--epoch', required = False, default = 0, type = int)

    # parser.add_argument('--start_epoch', required = False, default = 0, type = int)
    parser.add_argument('--kfold', required = False, default = 0)
    # parser.add_argument('--labels_path', required = False, default = "/sciclone/geograd/heather_data/ti/data/central_asia_ys_target.json")
#     parser.add_argument('--labels_path', required = False, default = "/sciclone/geograd/Heather/c1/data/clean/c2_ys.json")
    parser.add_argument('--transforms_path', required = False, default = "/sciclone/geograd/Heather/c1/transform_stats_sc.json")
    parser.add_argument('--results_dir', required = False, default = "/sciclone/geograd/heather_data/imprecision/schools/models/")
    parser.add_argument('--target_name', required = False, default = "target")
    parser.add_argument('--device', required = False, default = "cuda")
    parser.add_argument('--postval', action='store_true', help='')
    parser.add_argument('--use_means', action='store_true', help='')
    parser.add_argument('--strict', action='store_false', help='')
    parser.add_argument('--norm_coords', action='store_true', help='')
    parser.add_argument('--use_lcs', action='store_true', help='')
    parser.add_argument('--nodups', action='store_true', help='')
    args = parser.parse_args() 
    
    print(args)

    transforms_path = args.transforms_path
    folder_name = os.path.join(args.results_dir, f"{args.region}_v{args.version}" )

    print(folder_name)

    if args.nodups:
        nodups = ""
    else:
        nodups = "_dup"

    labels_path = f"/sciclone/geograd/heather_data/imprecision/schools/data/{args.region}{nodups}_ys.json"
    coords_path = f"/sciclone/geograd/heather_data/imprecision/schools/data/{args.region}{nodups}_coords.json"
    lc_path = f"/sciclone/geograd/heather_data/imprecision/schools/data/{args.region}{nodups}_lcs.json"
    model_name = args.model_name
            
    # Load transformation stats
    with open(transforms_path, "r") as f:
        tstats = json.load(f)    
    ts = apply_transforms(tstats)
        
    device = args.device

    if not os.path.exists(f"{folder_name}/kfold{args.kfold}/{args.target_name}_results/"):
        os.mkdir(f"{folder_name}/kfold{args.kfold}/{args.target_name}_results/")

    with open(f"{folder_name}/kfold{args.kfold}/records.txt", "r") as f:
        stats = f.read().splitlines()
    v_stats = [float(i.split(": ")[-1]) for i in stats if "test" in i]#[0:145]
    print(np.min(v_stats), np.argmin(v_stats), len(v_stats))

    epoch = np.argmin(v_stats)

    # Instantiate model
    model = construct_model(args.model_name, args.norm_coords, args.use_means, args.use_lcs)
    model.to(args.device)          
    
    weights = torch.load(f"{folder_name}/kfold{args.kfold}/most_recent_epoch{epoch}.torch")["model_state_dict"]
    model.load_state_dict(weights, strict = args.strict)

    if args.eval:
        model.eval();
    
    with open(labels_path, "r") as file:
        labels = json.load(file)
    labels = list(labels.keys())
    print(labels[0:5])

    # Initialize datasets and transform
    target_dataset = PlanetData(labels_path = labels_path, 
                                coords_path = coords_path,
                                lc_path = lc_path,
                                transform = ts, 
                                # sample = args.sample, 
                                postval = True,
                                lc = args.use_lcs)
    
    preds, labs, ns = [], [], []

    for count, (inputs, targets, coords, imname) in enumerate(target_dataset):
    
        if (args.model_name in ["SpA", "FC"]) or (args.norm_coords):
            coords = coords.unsqueeze(0)

        if args.model_name in ["SpA", "GeoConv", "FC"]:
            output = model(inputs.unsqueeze(0).to(device), coords.to(device))
        else:
            output = model(inputs.unsqueeze(0).to(device))
    
        preds.append(output.item())
        labs.append(targets)
        ns.append(imname)
        print(count, len(target_dataset), end = "\r")
    
        if count % 2500 == 0:
            df = pd.DataFrame([preds, labs, ns]).T
            df.columns = ["pred", "label", "name"]
            df.to_csv(f"{folder_name}/kfold{args.kfold}/{args.target_name}_results/epoch{epoch}_target{nodups}_preds_new.csv", index = False) 

    
    
    df = pd.DataFrame([preds, labs, ns]).T
    df.columns = ["pred", "label", "name"]
    df.to_csv(f"{folder_name}/kfold{args.kfold}/{args.target_name}_results/epoch{epoch}_target{nodups}_preds_new.csv", index = False)
    
    print(r2_score(df["label"], df["pred"]))



