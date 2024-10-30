import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from copy import deepcopy




da = True

# # v2 kFold: 0
# nodups_preds = "./models/phl_v2/kfold0/results/epoch101_preds.csv"
# dups_preds = "./models/phl_v2/kfold0/phl_results/epoch101_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v2/kfold0/phl_results/dup_tsne_result_1d_p10.json"

# # v2 kFold: 1
# nodups_preds = "./models/phl_v2/kfold1/results/epoch93_preds.csv"
# dups_preds = "./models/phl_v2/kfold1/phl_results/epoch93_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v2/kfold1/phl_results/dup_tsne_result_1d_p10.json"

# # v2 kFold: 2
# nodups_preds = "./models/phl_v2/kfold2/results/epoch99_preds.csv"
# dups_preds = "./models/phl_v2/kfold2/phl_results/epoch99_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v2/kfold2/phl_results/dup_tsne_result_1d_p10.json"



# v2 kFold: 0
# nodups_preds = "./models/phl_v10/kfold0/results/epoch197_preds.csv"
# dups_preds = "./models/phl_v10/kfold0/phl_results/epoch197_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v10/kfold0/phl_results/dup_tsne_result_1d_p10.json"

# v2 kFold: 1
# nodups_preds = "./models/phl_v10/kfold1/results/epoch192_preds.csv"
# dups_preds = "./models/phl_v10/kfold1/phl_results/epoch192_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v10/kfold1/phl_results/dup_tsne_result_1d_p10.json"


# # v2 kFold: 2
# nodups_preds = "./models/phl_v10/kfold2/results/epoch192_preds.csv"
# dups_preds = "./models/phl_v10/kfold2/phl_results/epoch192_target_dup_preds_new.csv"
# fcs_path = "./models/phl_v10/kfold2/phl_results/dup_tsne_result_1d_p10.json"


# v7 kFold: 0
nodups_preds = "./models/phl_v7/kfold0/results/epoch133_preds.csv"
dups_preds = "./models/phl_v7/kfold0/phl_results/epoch133_target_dup_preds_new.csv"
fcs_path = "./models/phl_v7/kfold0/phl_results/dup_tsne_result_1d_p10.json"

# v7 kFold: 1
nodups_preds = "./models/phl_v7/kfold1/results/epoch91_preds.csv"
dups_preds = "./models/phl_v7/kfold1/phl_results/epoch91_target_dup_preds_new.csv"
fcs_path = "./models/phl_v7/kfold1/phl_results/dup_tsne_result_1d_p10.json"

# v7 kFold: 2
nodups_preds = "./models/phl_v7/kfold2/results/epoch51_preds.csv"
dups_preds = "./models/phl_v7/kfold2/phl_results/epoch51_target_dup_preds_new.csv"
fcs_path = "./models/phl_v7/kfold2/phl_results/dup_tsne_result_1d_p10.json"



combos = [
          [True, True, True],
          [True, False, False],
          [False, True, True],
          [False, False, True],
          [False, True, False],
          [True, False, True],
          [True, True, False],
         ]



for combo in combos:

    use_coords, use_stats, use_fc = combo[0], combo[1], combo[2]

    print("Use coords: ", combo[0])
    print("Use stats: ", combo[1])
    print("Use fc: ", combo[2])

    if da == True & use_fc == True:
        print("\n")
        continue

    # Function to calculate r2 and RMSE
    def r2_rmse_og(g):
        r2 = r2_score(g['mean_true_y'], g['avg_pred_y'])
        rmse = np.sqrt(mean_absolute_error(g['mean_true_y'], g['avg_pred_y']))
        return pd.Series(dict(r2=r2, rmse=rmse))
    
    # Load labels, coordinates, and train labels
    with open("/sciclone/geograd/heather_data/imprecision/schools/data/phl_dup_ys.json", "r") as f:
        labels = json.load(f)
    
    with open("/sciclone/geograd/heather_data/imprecision/schools/data/phl_dup_coords.json", "r") as f:
        coords = json.load(f)

    if da == False:
    
        with open(fcs_path, "r") as f:
            fcs = json.load(f)
    
    # Load labels, coordinates, and train labels
    with open("/sciclone/geograd/heather_data/imprecision/schools/data/phl_ys.json", "r") as f:
        labels2 = json.load(f)
    
    with open("/sciclone/geograd/heather_data/imprecision/schools/data/phl_coords.json", "r") as f:
        coords2 = json.load(f)
    
    coords.update(coords2)
    labels.update(labels2)
    
    if da != True:
        with open(fcs_path, "r") as f:
            fcs = json.load(f)
    
    # Load validation IDs
    val_ids = pd.read_csv(nodups_preds)
    val_ids["DHSID"] = val_ids["name"].str.split("/").str[-1].str.split("_").str[0].str.split(".").str[0]
    val_ids = val_ids["DHSID"].unique().tolist()
    
    # Load and prepare the predictions data
    def prepare_df(filepath):
        df = pd.read_csv(filepath)
        df["label"] = df["name"].map(labels)
        df["DHSID"] = df["name"].str.split("/").str[-1].str.split("_").str[0].str.split(".").str[0]
        df["coords"] = df["name"].map(coords)
        df["lon"] = df["coords"].str[0]
        df["lat"] = df["coords"].str[1]

        if da != True:
            df["fc"] = df["name"].map(fcs)
            df["fc"] = df["fc"].str[0].astype(float)
        
        return df
    
    df = prepare_df(dups_preds)
    df2 = prepare_df(nodups_preds)
    df = pd.concat([df, df2], ignore_index=True)
    # print(df.shape)
    df = df.dropna()
    # print(df.shape)
    
    # Split into training and validation sets based on val_ids
    train_df = df[~df["DHSID"].isin(val_ids)].copy()
    test_df = df[df["DHSID"].isin(val_ids)].copy()

    test_df = test_df.dropna()

    # print(train_df.head())

    # print(train_df.shape)
    # print(test_df.shape)

    # train_df = train_df.dropna()
    # train_df = train_df.dropna()
    
    # Prepare X_train, y_train, X_test, y_test
    def prepare_features_targets(df):
        df["error"] = df["label"] - df["pred"]
        df['group_std_dev'] = df.groupby('DHSID')['pred'].transform('std')
        df['individual_deviation'] = df['pred'] - df.groupby('DHSID')['pred'].transform('mean')
        # df['individual_deviation'] = abs(df['individual_deviation'])
        df['group_mean'] = df.groupby('DHSID')['pred'].transform('mean')
        df = df.dropna()
    
        # Initialize an empty list to store selected features
        features = []
        
        # Add statistical features if use_stats is True
        if use_stats:
            features.append(df['group_std_dev'].values.reshape(-1, 1))
            features.append(df['individual_deviation'].values.reshape(-1, 1))
            features.append(df['group_mean'].values.reshape(-1, 1))
        
        # Add coordinate features if use_coords is True
        if use_coords:
            features.append(df['lon'].values.reshape(-1, 1))
            features.append(df['lat'].values.reshape(-1, 1))
        
        # Add feature embeddings if use_fc is True
        if use_fc:
            features.append(df['fc'].values.reshape(-1, 1))
        
        # Stack the selected features horizontally to form X
        if features:
            X = np.hstack(features)
        else:
            raise ValueError("No features selected for the feature matrix.")
            
        y = df['error'].values
        return X, y, df["name"].to_list()
    
    # Prepare training and testing data
    X_train, y_train, train_names = prepare_features_targets(train_df)
    X_test, y_test, test_names = prepare_features_targets(test_df)

    test_df = test_df[test_df["name"].isin(test_names)]
    
    # Train model
    model = RandomForestRegressor(max_depth=10, min_samples_split=2, max_features='sqrt', min_samples_leaf=4)
    model.fit(X_train, y_train)
    
    # Validate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate R2 scores for train and test
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # print(f"Train R2: {train_r2}")
    # print(f"Test R2: {test_r2}")
    
    # Adjust predictions on test set
    test_df["pred_loss"] = y_test_pred
    test_df["adjusted_pred"] = test_df["pred"] + test_df["pred_loss"]

    test_df["id"] = test_df["name"].str.split("/").str[-1].str.split(".").str[0]
    val_ids = pd.read_csv(nodups_preds)
    val_ids["id"] = val_ids["name"].str.split("/").str[-1].str.split(".").str[0]
    val_ids = list(val_ids["id"].unique())
    # print(val_ids[0:5])
    # print(len(val_ids))
    test_df = test_df[test_df["id"].isin(val_ids)]
    test_df = test_df.drop_duplicates(subset = "id")


    test_df = test_df[test_df['name'].isin(df2["name"])]

    # Final R2 on test data with adjusted predictions
    adjusted_r2 = r2_score(test_df["label"], test_df["adjusted_pred"])
    og_r2 = r2_score(test_df["label"], test_df["pred"])

    print(test_df.shape)
    print(f"OG Test R2: {og_r2}")
    print(f"Adjusted Test R2: {adjusted_r2}", "\n\n")