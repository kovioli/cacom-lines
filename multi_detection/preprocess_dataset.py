import os
import json
from types import new_class
import shutil
import numpy as np
import cv2
#import imgaug as ia
#from imgaug import augmenters as iaa
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path


class PreprocessDataset():
    """
    Class to preprocess images suitable for the DataGenerator (training)
    Reads in images and annotations, reshapes both of them (cutting + padding), and saves in formatted_img_dir and formatted_annot_dir.
    Finally splits the data to train, validation and tests sets.
    """

    def __init__(self, im_size = 640):
        self.image_dir = 'data/raw/images/'
        self.annot_dir = 'data/raw/annotations/'
        self.formatted_img_dir = 'data/formatted/images/'
        self.formatted_annot_dir = 'data/formatted/annotations/'
        self.image_size = im_size
        self.init_folders()
        

    def init_folders(self):
        print("asdf")
        folders = ['data/formatted/images','data/formatted/annotations', 'data/train_set/images',
                   'data/train_set/annotations', 'data/validation_set/images', 'data/validation_set/annotations', 
                   'data/test_set/images', 'data/test_set/annotations']
        for f_name in folders:
            Path(f_name).mkdir(parents=True, exist_ok=True)


    def get_poly(self, annot_path):
        """funciton used by the preprocessing (at the data conversion)"""
        with open(annot_path) as handle:
            data = json.load(handle)

        shape_dicts = data['points']
        return shape_dicts


    def create_label2poly(self, cls, poly):
        label2poly = dict((el, []) for el in cls)
        for i, item in enumerate(cls):
            label2poly[item].append(poly[i])
        return label2poly


    def sorted_files(self, dir):
        """Sorts the files in a given directory

        Args:
            dir (str): directory path to be sorted

        Returns:
            list: list of sorted elements in a directory
        """
        return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0].split('_')[1]))


    def pad_resize(self, im, shape_dicts):
        vertical_cut = 250
        horizontal_cut = [80, 150]

        #image padding + resize:
        im = im[vertical_cut:-vertical_cut,horizontal_cut[0]:-horizontal_cut[1]]
        height, width, cc = im.shape
        side = max(height, width)

        ratio =  self.image_size / side
        
        black = (0, 0, 0)
        result = np.full((side, side, cc), black, dtype=np.uint8)

        xx = (side - width) // 2
        yy = (side - height) // 2

        result[yy:yy+height, xx:xx+width] = im
        result = cv2.resize(result, (self.image_size, self.image_size))

        shift = max(xx, yy)

        #annotation padding + resize:
        for shape in shape_dicts:
            for points in shape:
                if height > width: # points[0] is the X coordinate
                    points[0] = round((points[0] - horizontal_cut[0] + shift) * ratio)
                    points[1] = round((points[1] - vertical_cut)*ratio)
                else:
                    points[1] = round((points[1] - vertical_cut + shift) * ratio)
                    points[0] = round((points[0] - horizontal_cut[0])*ratio)

        return result, shape_dicts


    def show_im_mask(self, im, shape_dicts):
        mask = self.draw_multi_masks(im, shape_dicts)
        horiz = np.concatenate((im, mask), axis = 1)
        cv2.imshow("image", horiz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def save_img_annot(self, im, shape_dicts, img_path, annot_path):
        img_path = img_path.replace("raw", "formatted")
        annot_path = annot_path.replace("raw", "formatted")

        
        #save image
        #print("Image saved under: " + img_path)
        im_save = Image.fromarray(im)
        im_save.save(img_path)

        #save annotation:
        #print("Image saved under: " + annot_path)
        with open(annot_path, 'w') as f:
            json.dump(shape_dicts, f)


    def split_data(self, nr_test = 20, valid_ratio = 0.2):
        image_paths = [os.path.join(self.formatted_img_dir, x) for x in sorted(os.listdir(self.formatted_img_dir)) if x.split('.')[-1] == 'png']        
        annot_paths = [os.path.join(self.formatted_annot_dir, x) for x in sorted(os.listdir(self.formatted_annot_dir)) if x.split('.')[-1] == 'json']

        #random.shuffle(image_paths)
        paths = list(zip(image_paths, annot_paths))

        random.shuffle(paths)

        
        nr_val = (len(paths)-nr_test) // (1/valid_ratio)
        
        for path in paths[:int(nr_test)]:
            [shutil.copyfile(path[x], path[x].replace("formatted", "test_set")) for x in range(2)]

        for path in paths[int(nr_test):int(nr_test+nr_val)]:
            [shutil.copyfile(path[x], path[x].replace("formatted", "validation_set")) for x in range(2)]
            #shutil.copyfile(path[1], path[1].replace("formatted", "validation_set"))

        for path in paths[int(nr_test + nr_val):]:
            [shutil.copyfile(path[x], path[x].replace("formatted", "train_set")) for x in range(2)]
            #shutil.copyfile(path[1], path[1].replace("formatted", "train_set"))

        return nr_test, (len(paths)-nr_val-nr_test), nr_val

    def json_shape_dicts(self, annot_path):
        with open(annot_path) as handle:
            shape_dicts = json.load(handle)
        return shape_dicts


    def preprocess_dataset(self):
        #annot_names = [x for x in os.listdir(folder_name) if x.split('.')[-1] == 'json']
        image_paths = [os.path.join(self.image_dir, x) for x in sorted(os.listdir(self.image_dir)) if x.split('.')[-1] == 'png']
        annot_paths = [os.path.join(self.annot_dir, x) for x in sorted(os.listdir(self.annot_dir)) if x.split('.')[-1] == 'json']

        for i, (img_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
            """READ IN IMAGE AND SHAPES (LINES)"""
            im = cv2.imread(img_path, 1) #reads image as RGB
            shape_dicts = self.get_poly(annot_path)
            
            """PAD + RESIZE IMAGES AND ANNOTATIONS"""
            im, shape_dicts = self.pad_resize(im, shape_dicts)
            
            self.save_img_annot(im, shape_dicts, img_path, annot_path)

        """SPLIT IMAGES (TRAIN + VALIDATION)"""
        nr_test, nr_train, nr_val = self.split_data()
        return nr_test, nr_train, nr_val

       
def main():
    preprocessor = PreprocessDataset()
    nr_test, nr_train, nr_val = preprocessor.preprocess_dataset()
    print("Nr of test data: " + str(nr_test))
    print("Nr of training data: " + str(nr_train))
    print("Nr of validation data: " + str(nr_val))

if __name__ == "__main__":
    main()
