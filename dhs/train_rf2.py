import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from copy import deepcopy




da = False


# for fold in [0,1,2]:

# v7 kFold: 0
nodups_preds = "./new_models/western_africa_v7/kfold0/results/epoch188_nodups_preds.csv"
dups_preds = "./new_models/western_africa_v7/kfold0/western_africa_results/epoch188_target_preds.csv"
fcs_path = "./new_models/western_africa_v7/kfold0/western_africa_results/dup_tsne_result_2d_p45.json"

print(fcs_path)

# v7 kFold: 1
# nodups_preds = "./new_models/western_africa_v7/kfold1/results/epoch177_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v7/kfold1/western_africa_results/epoch177_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v7/kfold1/western_africa_results/tsne_result_1d_p10.json"

# v7 kFold: 2
# nodups_preds = "./new_models/western_africa_v7/kfold2/results/epoch178_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v7/kfold2/western_africa_results/epoch178_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v7/kfold2/western_africa_results/tsne_result_1d_p10.json"


# v10 kFold: 0
# nodups_preds = "./new_models/western_africa_v10/kfold2/results/epoch76_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v10/kfold2/western_africa_results/epoch76_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v10/kfold2/western_africa_results/tsne_result_1d_p10.json"

# v10 kFold: 1
# nodups_preds = "./new_models/western_africa_v10/kfold1/results/epoch177_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v10/kfold1/western_africa_results/epoch177_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v10/kfold1/western_africa_results/tsne_result_1d_p10.json"

# v10 kFold: 2
# nodups_preds = "./new_models/western_africa_v10/kfold2/results/epoch178_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v10/kfold2/western_africa_results/epoch178_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v10/kfold2/western_africa_results/tsne_result_1d_p10.json"



# v7 kFold: 0
# nodups_preds = "./new_models/western_africa_v12/kfold0/results/epoch116_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v12/kfold0/western_africa_results/epoch116_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v12/kfold0/western_africa_results/tsne_result_1d_p10.json"

# v7 kFold: 1
# nodups_preds = "./new_models/western_africa_v7/kfold1/results/epoch177_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v7/kfold1/western_africa_results/epoch177_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v7/kfold1/western_africa_results/tsne_result_1d_p10.json"

# v7 kFold: 2
# nodups_preds = "./new_models/western_africa_v7/kfold2/results/epoch178_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v7/kfold2/western_africa_results/epoch178_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v7/kfold2/western_africa_results/tsne_result_1d_p10.json"




# nodups12 kFold: 0
# nodups_preds = "./new_models/western_africa_vnodups12/kfold0/results/epoch29_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_vnodups12/kfold0/western_africa_results/epoch29_target_preds.csv"
# if da == True:
#     fcs_path = "./new_models/western_africa_vnodups12/kfold0/western_africa_results/tsne_result_1d_p10.json"

# nodups12 kFold: 1
# nodups_preds = "./new_models/western_africa_vnodups12/kfold1/results/epoch63_preds.csv"
# dups_preds = "./new_models/western_africa_vnodups12/kfold1/western_africa_results/epoch63_target_preds.csv"
# if da == True:
#     fcs_path = "./new_models/western_africa_vnodups12/kfold1/western_africa_results/tsne_result_1d_p10.json"

# # nodups12 kFold: 2
# nodups_preds = "./new_models/western_africa_vnodups12/kfold2/results/epoch61_preds.csv"
# dups_preds = "./new_models/western_africa_vnodups12/kfold2/western_africa_results/epoch61_target_preds.csv"
# if da == True:
#     fcs_path = "./new_models/western_africa_vnodups12/kfold2/western_africa_results/tsne_result_1d_p10.json"




# v12 kFold: 0
# nodups_preds = "./new_models/western_africa_v12/kfold0/results/epoch116_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v12/kfold0/western_africa_results/epoch116_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v12/kfold0/western_africa_results/tsne_result_1d_p10.json"

# v12 kFold: 1
# nodups_preds = "./new_models/western_africa_v12/kfold1/results/epoch186_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v12/kfold1/western_africa_results/epoch186_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v12/kfold1/western_africa_results/dup_tsne_result_1d_p10.json"

# v7 kFold: 2
# nodups_preds = "./new_models/western_africa_v7/kfold2/results/epoch178_nodups_preds.csv"
# dups_preds = "./new_models/western_africa_v7/kfold2/western_africa_results/epoch178_target_preds_new.csv"
# fcs_path = "./new_models/western_africa_v7/kfold2/western_africa_results/tsne_result_1d_p10.json"

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
    with open("/sciclone/geograd/heather_data/ti/data_new/western_africa_ys.json", "r") as f:
        labels = json.load(f)
    
    with open("/sciclone/geograd/heather_data/ti/data_new/western_africa_coords.json", "r") as f:
        coords = json.load(f)

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
    # df2 = prepare_df(nodups_preds)
    # df = pd.concat([df, df2])

    # print(df.shape)
    
    # Split into training and validation sets based on val_ids
    train_df = df[~df["DHSID"].isin(val_ids)].copy()    
    test_df = df[df["DHSID"].isin(val_ids)].copy()

    # print("train df: ", train_df.shape)
    # print("test df: ", test_df.shape)
    
    # Prepare X_train, y_train, X_test, y_test
    def prepare_features_targets(df, train = False):
        df["error"] = df["label"] - df["pred"]
        df['group_std_dev'] = df.groupby('DHSID')['pred'].transform('std')
        df['individual_deviation'] = df['pred'] - df.groupby('DHSID')['pred'].transform('mean')
        # df['individual_deviation'] = abs(df['individual_deviation'])
        df['group_mean'] = df.groupby('DHSID')['pred'].transform('mean')

        df.to_csv(f"./test_{use_coords}_{use_stats}_{use_fc}.csv", index = False)

        # print("1: ", df.shape)

        # if train:
        #     df = df[df["name"].str.endswith("_1.tiff")]


        # print(df["name"][0])

        # print("2: ", df.shape)

        # print(df.head())
        
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

        print(X.shape)
            
        y = df['error'].values
        return X, y
    
    # Prepare training and testing data
    X_train, y_train = prepare_features_targets(train_df, train = True)
    X_test, y_test = prepare_features_targets(test_df)
    
    # Train model
    model = RandomForestRegressor(max_depth = 14, min_samples_split = 3, max_features='sqrt', min_samples_leaf = 3)
    # model = RandomForestRegressor(max_depth = 14, min_samples_split = 3, max_features='sqrt', min_samples_leaf = 3)
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

    # Final R2 on test data with adjusted predictions
    adjusted_r2 = r2_score(test_df["label"], test_df["adjusted_pred"])
    og_r2 = r2_score(test_df["label"], test_df["pred"])
    
    
    print(test_df.shape)
    
    print(f"OG Test R2: {og_r2}")
    print(f"Adjusted Test R2: {adjusted_r2}", "\n\n")

    # import eli5
    # from eli5.sklearn import PermutationImportance    


    # if use_coords == True & use_stats == True & use_fc == False:
        
    #     # Run PFI on the model
    #     perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
        
    #     # Display feature importance
    #     eli5.show_weights(perm, feature_names=['group_std_dev', 'individual_deviation', 'group_mean', 'lon', 'lat', 'fc'])