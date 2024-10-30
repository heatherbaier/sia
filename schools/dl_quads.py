import geopandas as gpd
import pandas as pd
import requests
import urllib
import json
import os


#setup API KEY
PLANET_API_KEY = '' # <= insert API key here 
#setup Planet base URL
API_URL = "https://api.planet.com/basemaps/v1/mosaics"
#setup session
session = requests.Session()
#authenticate
session.auth = ("PLAKd25e5a5054894a8a9fb73cfb9e52e3e3", "") #<= change to match variable for API Key if needed


gdf = gpd.read_file("./data/phl_random_points_2km.shp")
gdf["val"] = 1
gdf['cumsum'] = gdf[["school_id", 'val']].groupby('school_id').cumsum()
gdf = gdf.rename(columns = {"cumsum": "cs"})
gdf = gdf.set_crs("EPSG:4326")
print(gdf.head())
print(gdf.shape)


# gdf = gpd.read_file("./data/planet_buffers4.shp")
# gdf = gdf.set_crs("EPSG:4326")
# print(gdf.head())
# print(gdf.shape)


# laksdglksal


base_dir = "./imagery/quads_2km/"
# quad_tracker = {}
# downloaded = []


downloaded = os.listdir(base_dir)
print("DOWNLOADED: ", downloaded[0:5])


# with open("../quads_tracker.json", "r") as f:
#     quad_tracker = json.load(f)
    
quad_tracker = {}


c = 0
for col, row in gdf.iterrows():
    
    # try:
    
    bounds = row.geometry.bounds

    # mosaic = f"global_quarterly_2023q3_mosaic"

    mosaic = "planet_medres_visual_2022-12_mosaic"

    #set params for search using name of mosaic
    parameters = {
        "name__is": mosaic # <= customize to your use case
    }

    #make get request to access mosaic from basemaps API
    res = session.get(API_URL, params = parameters)

    #response status code
#     print(res.status_code)

    mosaic = res.json()

    #get id
    mosaic_id = mosaic['mosaics'][0]['id']

    #get bbox for entire mosaic
    mosaic_bbox = bounds

    #converting bbox to string for search params
    string_bbox = ','.join(map(str, mosaic_bbox))

#     print('Mosaic id: '+ mosaic_id)
#     print('Mosaic bbox: '+ string_bbox)

    #search for mosaic quad using AOI
    search_parameters = {
        'bbox': string_bbox,
        'minimal': True
    }

    #accessing quads using metadata from mosaic
    quads_url = "{}/{}/quads".format(API_URL, mosaic_id)
    res = session.get(quads_url, params = search_parameters, stream = True)
#     print(res.status_code)   

    quads = res.json()
    items = quads['items']

    #printing an example of quad metadata
#     print(len(items))  

    #iterate over quad download links and save to folder by id
    quad_ids = []
    for i in items:
        link = i['_links']['download']
        name = i['id']
        quad_ids.append(name)
        name = name + '.tiff'
        filename = os.path.join(base_dir, name)

        if name not in downloaded:

            #checks if file already exists before s
            if not os.path.isfile(filename):
                urllib.request.urlretrieve(link, filename)  

        downloaded.append(name)

    k = str(row.school_id) + "_" + str(row.cs)

    quad_tracker[k] = quad_ids
        
    # except:
        
    #     pass        
        
        
    c += 1

    print(c, end = "\r")

    if c % 50:
        with open("./quads_tracker_phl_dups.json", "w") as f:
            json.dump(quad_tracker, f)

    with open("./dl_counter_phl_dups.txt", "w") as f:
        f.write(str(c) + " / " + str(len(gdf)))


        