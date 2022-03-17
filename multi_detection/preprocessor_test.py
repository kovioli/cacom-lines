import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Tests whether the the funciton preprocess_dataset.py functioned properly (for visual inspection)
"""

formatted_img_dir = 'data/formatted/images/'
image_paths = [os.path.join(formatted_img_dir, x) for x in sorted(os.listdir(formatted_img_dir))]


formatted_annot_dir = 'data/formatted/annotations/'
annot_paths = [os.path.join(formatted_annot_dir, x) for x in sorted(os.listdir(formatted_annot_dir))]

#np. set_printoptions(threshold=np. inf)

for i, (img_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
    with open(annot_path) as handle:
        data = json.load(handle)

    #blank = np.zeros(shape=(640,640, 3), dtype=np.uint8)
    channels = []

    im = cv2.imread(img_path)

    for i, lines in enumerate(data):
        blank = np.zeros(shape=(640,640,3), dtype=np.uint8)

        for point1, point2 in zip(lines, lines[1:]):
            if i == 0:
                cv2.line(blank, point1, point2, (51, 255, 51), 2)
            else:
                cv2.line(blank, point1, point2, (230, 12, 200), 2)
            #cv2.line(blank, point1, point2, 1, 2)
            #cv2.line(blank, point1, point2, 1, 2)
        channels.append(blank)

    
    new_im = channels[0] + channels[1]
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)

    cv2.imshow('annotation', new_im)
    cv2.waitKey(0)
