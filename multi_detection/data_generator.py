import numpy as np
import pickle
import os
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import json
import tensorflow as tf
from config import imshape, n_classes




sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# ia.seed(1)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.3), # horizontally flip 50% of all images
        iaa.Flipud(0.1), # vertically flip 20% of all images
     
        sometimes(iaa.Affine(
                scale = {"x": (0.8, 1.2), "y": (0.8, 1.2)},
                shear = (-15, 15),
                rotate = (-40, 40)
        )),
        # execute 0 to 3 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 3 and 11
                ]),
                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0)), # emboss images
             
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
             
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
            
                sometimes(iaa.PerspectiveTransform(scale=(0.0, 0.08)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, annot_paths, batch_size=32,
                 shuffle=True, augment=False, filter=False):
        """Creates a DataGenerator object suited for tensorflow. It feeds the data to the training, 
        generating it each epoch. It makes the data customizable (such as augmentation), and uses 
        way less memory (loading in just parts, not the whole of the dataset)
        Args:
            image_paths (list[str]): Path to the images
            annot_paths (list[str]): Path to the annotations
            batch_size (int, optional): Batch size of the training. Defaults to 32.
            shuffle (bool, optional): Defines whether the data will be shuffled during training. Defaults to True.
            augment (bool, optional): Defines whether the data should be augmented during training. Defaults to False.
        """
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.filter = filter
        self.on_epoch_end()


    def __len__(self):
        """Calculates the number of batches per epoch
        Returns:
            int: the number of batches per epoch 
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))


    def __getitem__(self, index):
        """Generates indexes of the batch, and feeds the paths to the __data_generation fn
        Args:
            index (int): index of the batch inside the epoch
        Returns:
            np.array, np.array: Input and output matrixes
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)

        return X, y


    def on_epoch_end(self):
        """Updates (shuffles) the indexes at the end of each epoch
        """
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def create_multi(self, im, shape_dicts):
        """
        updated fn for multiclass cacom
        """

        channels = []
        background = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        
        for lines in shape_dicts:
            blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)

            for point1, point2 in zip(lines, lines[1:]):
                cv2.line(blank, tuple(round(x) for x in point1), tuple(round(x) for x in point2), 255, 2)
                cv2.line(background, tuple(round(x) for x in point1), tuple(round(x) for x in point2), 255, 2)
        
            #blank = blank / 255.0
            channels.append(blank)

        _, background = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
        channels.append(background)

        Y = np.stack(channels, axis=2) / 255.0
        return Y


    def json_shape_dicts(self, annot_path):
        """Reads in data from a json data
        Args:
            annot_path (str): path of the annotation
        Returns:
            dict: dictionary containing the shapes
        """
        with open(annot_path) as handle:
            shape_dicts = json.load(handle)
        return shape_dicts


    def augment_poly(self, im, shape_dicts):
        """Augments the images and the corresponding annotations
        Args:
            im (np.array): image as numpy array
            shape_dicts (dict): the original dictionary containing the shapes
        Returns:
            np.array, dict: the augmented image and dictionary
        """
        # augments an image and it's polygons

        points = []
        aug_shape_dicts = []
        i = 0

        for line in shape_dicts:
            for coord in line:
                points.append(ia.Keypoint(x=coord[0], y=coord[1]))

            _d = {}
            _d['index'] = (i, i+len(line))
            aug_shape_dicts.append(_d)

            i += len(line)

        keypoints = ia.KeypointsOnImage(points, shape=(imshape[0], imshape[1]))

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([im])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]


        return_shape_dicts = []
        for shape in aug_shape_dicts:
            [start, end] = shape['index']
            aug_points = [[keypoint.x, keypoint.y] for keypoint in keypoints_aug.keypoints[start:end]]
            return_shape_dicts.append(aug_points)

        return image_aug, return_shape_dicts


    def create_mask(self, im, data):
        blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        
        for lines in data:
            for point1, point2 in zip(lines, lines[1:]):
                cv2.line(blank, tuple(round(x) for x in point1), tuple(round(x) for x in point2), 255, 2)
        
        blank = blank / 255.0
    
        return np.expand_dims(blank, axis = 2)


    def laplacian_edge_detection(self, img, ksize):
        laplace = cv2.Laplacian(img,ddepth=cv2.CV_64F, ksize=ksize)
        return np.uint8(laplace)


    def gaussian_blur(self, img, size_kernel = (13,13), sigma=1):
        """Filter the image with a gaussian filter
            Args:
                size_kernel: (width,height) can differ but must be positive and odd
        """
        return cv2.GaussianBlur(img, (size_kernel[0], size_kernel[1]), sigma)


    def preprocess_filter(self, img, gauss_size=(5,5), gauss_sigma = 1, laplace_ksize = 3):
        # apply a gaussian filter to reduce noise
        img_blur = self.gaussian_blur(img, gauss_size, gauss_sigma)
        img_laplace = self.laplacian_edge_detection(img_blur, laplace_ksize)
        return img_laplace


    def __data_generation(self, image_paths, annot_paths):
        """Generates a batch of images and the corresponding masks
        Args:
            image_paths (str): path to the image
            annot_paths (str): path to the annotation
        Returns:
            np.array, np.array: The generated input and output used for the training
        """

        X = np.empty((self.batch_size, imshape[0], imshape[1], imshape[2]), dtype=np.float32)
        Y = np.empty((self.batch_size, imshape[0], imshape[1], n_classes),  dtype=np.float32)

        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
             # read image as grayscale or rgb
            if imshape[2] == 1:
                im = cv2.imread(im_path, 0)
                im = np.expand_dims(im, axis=2)
            elif imshape[2] == 3:
                im = cv2.imread(im_path, 1)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            

            shape_dicts = self.json_shape_dicts(annot_path)

            # check for augmentation
            if self.augment:
                im, shape_dicts = self.augment_poly(im, shape_dicts)
            # further preprocessing (filtering)
            if self.filter:
                im = self.preprocess_filter(im)

            # create target masks
            mask = self.create_multi(im, shape_dicts)

            #im = np.expand_dims(im, axis=2)
            X[i,] = im
            Y[i,] = mask
        
        return X, Y