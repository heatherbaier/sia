import geopandas as gpd
import numpy as np
import argparse

from data_utils import *


if __name__ == "__main__":
    
    gdf = gpd.read_file(f"./data/phl_voronoi.shp")
    gdf['random_points'] = gdf.apply(lambda x: gen_random_points(x['geometry'], 10), axis = 1)
    gdf = gdf[["DHSID", "DHSYEAR", "DHSCLUST", "folder", "buffer_siz", "hv001", "hv271", "geometry", "random_points"]]
    gdf = gdf.explode('random_points')
    gdf["val"] = 1
    gdf['cumsum'] = gdf[["DHSID", 'val']].groupby('DHSID').cumsum()
    gdf["lon"] = gdf["random_points"].astype(str).str.split(",").str[0].str.replace("(", "")
    gdf["lat"] = gdf["random_points"].astype(str).str.split(",").str[1].str.replace(")", "")
    gdf = gdf.drop(["hv001", "val", "random_points", "geometry"], axis = 1)

    gdf.to_csv(f"../data/clean/wealth_random_points.csv", index = False)