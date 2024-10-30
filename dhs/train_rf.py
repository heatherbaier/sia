from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from copy import deepcopy
import pandas as pd
import numpy as np
import json

def r2_rmse_og(g):
    r2 = r2_score(g['mean_true_y'], g['avg_pred_y'])
    rmse = np.sqrt(mean_absolute_error(g['mean_true_y'], g['avg_pred_y']))
    return pd.Series(dict(r2 = r2, rmse = rmse))


with open(f"/sciclone/geograd/heather_data/ti/data_new/western_africa_ys.json", "r") as f:
    labels = json.load(f)   

# with open(f"/sciclone/geograd/heather_data/ti/data_new/central_and_western_asia_ys_target.json", "r") as f:
with open(f"/sciclone/geograd/heather_data/ti/data_new/western_africa_coords.json", "r") as f:
    coords = json.load(f)  

with open(f"/sciclone/geograd/heather_data/ti/data_new/western_africa_ys_nodups.json", "r") as f:
    train_labels = json.load(f)  


val_ids = pd.read_csv("./new_models/western_africa_v7/kfold0/results/epoch188_nodups_preds.csv")
val_ids["DHSID"] = val_ids["name"].str.split("/").str[-1].str.split("_").str[0].str.split(".").str[0]
val_ids = list(val_ids["DHSID"].unique())
print(val_ids[0:5])



# Read predictions and assign labels
df = pd.read_csv("./new_models/western_africa_v7/kfold0/western_africa_results/epoch188_target_preds.csv")
df["label"] = df["name"].map(labels)
df["DHSID"] = df["name"].str.split("/").str[-1].str.split("_").str[0].str.split(".").str[0]
print("Dups shape: ", df.shape)

# Filter out validation IDs from the true dataset
true = df.copy()
true = true[~true["DHSID"].astype(str).isin(val_ids)]
print("R2 score: ", r2_score(true["label"], true["pred"]))
print("Non-dups shape:", true.shape)

# Mark true and duplicate data, combine both
true["t"] = 1
df["t"] = 0
df = df[df["DHSID"].isin(true["DHSID"])][true.columns]
df = pd.concat([true, df]).sort_values(by="DHSID")
print(df.shape)

# Group data by DHSID and calculate statistics
g = df.groupby("DHSID").agg(
    avg_pred_y=("pred", "mean"),
    med_pred_y=("pred", "median"),
    var_pred_y=("pred", "var"),
    std_pred_y=("pred", "std"),
    mean_true_y=("label", "mean"),
    var_true_y=("label", "var")
).reset_index()

# Add ISO and error calculations
g["iso"] = g["DHSID"].str[:2]
g["avg_error"] = abs(g["mean_true_y"] - g["avg_pred_y"])
g["med_error"] = abs(g["mean_true_y"] - g["med_pred_y"])
g = g.dropna()

# Map temp labels back to df
df["temp_lab"] = df["DHSID"].map(g.set_index("DHSID")["avg_pred_y"])
df.head()




test = deepcopy(df)
# test["iso"] = test["name"].str.split("/").str[10].str[0:2]
test["error"] = test["label"] - test["pred"]

test['group_std_dev'] = test.groupby('DHSID')['pred'].transform('std')
test['individual_deviation'] = test['pred'] - test['temp_lab']
test['group_mean'] = test.groupby('DHSID')['pred'].transform('mean')

print(test.shape)
test = test.dropna()
print(test.shape)

# Example features
group_std_dev = test['group_std_dev'].values  # standard deviations of predictions
individual_deviation = test['individual_deviation'].values  # individual deviations from group mean
group_mean = test['group_mean'].values  # group means of predictions

# Target epsilon (to estimate)
optimal_epsilons = test["error"].values

# Combine features
X = np.hstack([group_std_dev.reshape(-1, 1), 
               individual_deviation.reshape(-1, 1), 
               group_mean.reshape(-1, 1)])
               # fcs])  # Stack all features horizontally

y = optimal_epsilons

# Train the regression model
# model = RandomForestRegressor(max_depth = 12, min_samples_split = 2, max_features='sqrt', min_samples_leaf = 4)
model = RandomForestRegressor(max_depth = 10, min_samples_split = 2, max_features='sqrt', min_samples_leaf = 4)
# model = RandomForestRegressor(max_depth = 10, min_samples_split = 2, max_features='sqrt', min_samples_leaf = 3)
# model = RandomForestRegressor(max_depth = 10, min_samples_split = 4, max_features='sqrt', min_samples_leaf = 4)
model.fit(X, y)

print(r2_score(y, model.predict(X)))

test["pred_loss"] = model.predict(X)# * test["sign"]
test["adjusted_pred"] = test["pred"] + test["pred_loss"]
test.head()

r2_score(test["label"], test["adjusted_pred"])




# Read predictions and assign labels
df = pd.read_csv("./new_models/western_africa_v7/kfold0/western_africa_results/epoch188_target_preds.csv")
df["label"] = df["name"].map(labels)
df["DHSID"] = df["name"].str.split("/").str[-1].str.split("_").str[0].str.split(".").str[0]
print("Dups shape: ", df.shape)

# Filter in validation IDs from the true dataset
true = df.copy()
true = true[true["DHSID"].astype(str).isin(val_ids)]
print("R2 score: ", r2_score(true["label"], true["pred"]))
print("Non-dups shape:", true.shape)

# Mark true and duplicate data, combine both
true["t"] = 1
df["t"] = 0
df = df[df["DHSID"].isin(true["DHSID"])][true.columns]
# df = pd.concat([true, df]).sort_values(by="DHSID")
print(df.shape)

# Group data by DHSID and calculate statistics
g = df.groupby("DHSID").agg(
    avg_pred_y=("pred", "mean"),
    med_pred_y=("pred", "median"),
    var_pred_y=("pred", "var"),
    std_pred_y=("pred", "std"),
    mean_true_y=("label", "mean"),
    var_true_y=("label", "var")
).reset_index()

# Add ISO and error calculations
g["iso"] = g["DHSID"].str[:2]
g["avg_error"] = abs(g["mean_true_y"] - g["avg_pred_y"])
g["med_error"] = abs(g["mean_true_y"] - g["med_pred_y"])
g = g.dropna()

# Map temp labels back to df
df["temp_lab"] = df["DHSID"].map(g.set_index("DHSID")["avg_pred_y"])
df.head()



test = deepcopy(df)
# test["iso"] = test["name"].str.split("/").str[10].str[0:2]
test["error"] = test["label"] - test["pred"]

test['group_std_dev'] = test.groupby('DHSID')['pred'].transform('std')
test['individual_deviation'] = test['pred'] - test['temp_lab']
test['group_mean'] = test.groupby('DHSID')['pred'].transform('mean')

print(test.shape)
test = test.dropna()
print(test.shape)

# Example features
group_std_dev = test['group_std_dev'].values  # standard deviations of predictions
individual_deviation = test['individual_deviation'].values  # individual deviations from group mean
group_mean = test['group_mean'].values  # group means of predictions

# Fourth column 'fcs' containing 128 features
# fcs = np.array(train['fcs'].tolist())  # Convert list of lists to a NumPy array

# Target epsilon (to estimate)
optimal_epsilons = test["error"].values

# Combine features
X = np.hstack([group_std_dev.reshape(-1, 1), 
               individual_deviation.reshape(-1, 1), 
               group_mean.reshape(-1, 1)])
               # fcs])  # Stack all features horizontally

y = optimal_epsilons

print(r2_score(y, model.predict(X)))

test["pred_loss"] = model.predict(X)# * test["sign"]
test["adjusted_pred"] = test["pred"] + test["pred_loss"]
test.head()

r2_score(test["label"], test["adjusted_pred"])


print("New r2: ", r2_score(test["label"], test["adjusted_pred"]))
print("OG r2: ", r2_score(test["label"], test["pred"]))