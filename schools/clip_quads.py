from shapely.geometry import Polygon
from rasterio.merge import merge
from rasterio.plot import show
import rioxarray as riox
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import pandas as pd
import calendar    
import tarfile
import shapely
import json
import os


with open("./quads_tracker_phl_dups.json", "r") as f:
    quad_tracker = json.load(f)
# quad_tracker = dict(reversed(list(quad_tracker.items())))
    
# gdf = gpd.read_file("./data/planet_buffers4.shp")
# gdf = gdf.set_crs("EPSG:4326")
# print(gdf.head())
# print(gdf.shape)

gdf = gpd.read_file("./data/phl_random_points_buffers.shp")
gdf = gdf.rename(columns = {"cumsum": "cs"})
gdf["id"] = gdf["school_id"].astype(str) + "_" + gdf["cs"].astype(str)
gdf = gdf.set_crs("EPSG:4326")
print(gdf.head())
print(gdf.shape)

gdf = gdf.set_crs("EPSG:4326")
gdf = gdf.to_crs("epsg:3857")

quads_dir = "./imagery/quads_2km/"
save_dir = "./imagery/clips_2km/"


c = 0
# for k,v in quad_tracker.items():

for k, v in reversed(list(quad_tracker.items())):

    print(c, len(quad_tracker))

    iso = "PH"
    
    try:

        # if os.path.exists(os.path.join(quads_dir, v[0] + ".tiff")):
        #     print(k, v, " imagery exists")
        # else:
        #     print(k, v, " FAILED")

        if not os.path.exists(os.path.join(save_dir, iso)):
            os.mkdir(os.path.join(save_dir, iso))

        test_path = os.path.join(save_dir, iso, k + ".tiff")

        if not os.path.exists(test_path):
            
            # print(test_path, " doesn't exist! Clipping now!")


            if len(v) == 1:

                # print(os.path.join(quads_dir, v[0] + ".tiff"))

                raster = riox.open_rasterio(os.path.join(quads_dir, v[0] + ".tiff"))
                geom = gdf[gdf["id"] == k]["geometry"].squeeze()
                # print(geom)
                clipped_raster = raster.rio.clip([geom])
                clipped_raster.rio.to_raster(os.path.join(save_dir, iso, k + ".tiff"))  


            else:

                raster_to_mosiac = []
                for p in v:
                    raster = rio.open(os.path.join(quads_dir, p + ".tiff"))
                    raster_to_mosiac.append(raster)    

                mosaic, output = merge(raster_to_mosiac) 

                output_path = f"{save_dir}/{iso}/{k}_merged.tiff"

                # print(output_path)

                output_meta = raster.meta.copy()
                output_meta.update(
                    {"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": output,
                    }
                )        

                with rio.open(output_path, "w", **output_meta) as m:
                    m.write(mosaic)    

                # Read raster using rioxarray
                raster = riox.open_rasterio(output_path)

                # Shapely Polygon to clip raster
                geom = gdf[gdf["id"] == k]["geometry"].squeeze()

                # Use shapely polygon in clip method of rioxarray object to clip raster
                clipped_raster = raster.rio.clip([geom])

                # Save clipped raster
                clipped_raster.rio.to_raster(f"{save_dir}/{iso}/{k}.tiff")   

                os.remove(output_path)   


        else:
            print(test_path, " has already been clipped! Moving on...")


        c += 1



    except:

        pass