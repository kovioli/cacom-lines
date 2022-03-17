import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Convert the results.json file into separate .json files that can be effectively used for the training.
These are saved in folder_name, and image_folder is used for comparison (only creates .json if the corresponding image exists)

"""

folder_name = "data/raw/annotations"
image_folder = "data/raw/images"

im_names = [x.split('.')[0] for x in os.listdir(image_folder)]


with open('results.json') as handle:
    data = json.load(handle)

for block in data["Results block"]:
    json_dict = {
        "points": []
    }

    file_name = block[1].split("\\")[-1]

    if not file_name in im_names:
        #delete picture from raw
        #os.remove(os.path.join(image_folder, file_name + '.png'))
        continue

    for line in block[5][0]:
        json_dict["points"].append(line[2])
        
    with open(os.path.join(folder_name, file_name + ".json"), "w") as outfile:
        json.dump(json_dict, outfile)

annot_names = [x.split('.')[0] for x in os.listdir(folder_name)]

for im_name in im_names:
    if not im_name in annot_names:
        os.remove(os.path.join(image_folder, im_name + '.png'))

