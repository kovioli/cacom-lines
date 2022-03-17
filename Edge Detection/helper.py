"""This is a file containing classes and functions which are used multiple times"""
import matplotlib.pyplot as plt
import pydicom as dcm
import numpy as np
import cv2
import os
import pandas as pd
import datetime


def files(rel_path):
    """
    Input:  relative path to directory from main.py
    Return:  List of filepaths (compatible with read_dcm)
    """
    path_dicom = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    dcm_dir = os.listdir(path_dicom)
    return [os.path.join(path_dicom, file) for file in dcm_dir]


def img_to_greyscale(img):
    # Convert pixel_array (img) to -> gray image (img_2d_scaled)
    img = img.astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # scale
    img_2d_scaled = (np.maximum(imgGray,0) / imgGray.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)

    # alternative: however I believe that it is not scaled
    # img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_2d_scaled


def read_dcm(filepath):
    """read picture and return uint8 array"""
    ds = dcm.dcmread(filepath)
    img = ds.pixel_array
    # Rescaling grey scale between 0-255
    img_grey = img_to_greyscale(img)
    return img, img_grey


def plot_side_by_side(imgs,names,path=None, name = None):
    nr_of_images = len(imgs)
    n_rows = int(np.floor(np.sqrt(nr_of_images)))
    n_columns = int(np.ceil(np.sqrt(nr_of_images)))
    plt.figure(figsize=(10, 8))
    for i in range(nr_of_images):
        plt.subplot(n_rows,n_columns,i+1), plt.imshow(imgs[i], cmap = 'gray')
        plt.title(names[i]), plt.xticks([]), plt.yticks([])
    if path:
        plt.savefig(os.path.join(path, f"{name}.pdf"))
    plt.show()


class CannyEdgeDetection:
    def __init__(self, img):
        self.img = img
        self.img_blur = cv2.GaussianBlur(self.img, (3, 3), 0)
        self.docs = pd.DataFrame(columns = ["Thresh1", "Thresh2", "Blur"])
        cv2.imshow("Image",self.img_blur)
        cv2.namedWindow('canny')
        # self.k = cv2.waitKey(0) & 0xFF
        self.thresh1=100
        self.thresh2=1
        blur = 3

        cv2.createTrackbar('thresh1','canny',self.thresh1,255,self.funccan)
        cv2.createTrackbar('thresh2','canny',self.thresh2,255,self.funccan)
        cv2.createTrackbar('blur', 'canny', blur, 19, self.func_blur)

        self.func_blur(0)
        self.funccan(0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def funccan(self, thresh1=0):
        self.thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
        self.thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
        edge = cv2.Canny(self.img_blur, self.thresh1, self.thresh2)
        cv2.imshow('canny', edge)

    def func_blur(self, blur=0):
        self.blur = cv2.getTrackbarPos('blur', 'canny')
        if self.blur % 2 == 0:
            self.blur = self.blur + 1
        self.img_blur = cv2.GaussianBlur(self.img, (self.blur, self.blur), 0)
        self.funccan(0)

    def doc(self):
        self.docs["Thresh1"] = self.thresh1
        self.docs["Thresh2"] = self.thresh2
        self.docs["Blur"] = self.blur

        mydir = os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir)
        self.docs.to_csv(os.path.join(mydir, "parameters"))

    def auto_canny(self,path=None, sigma=0.33):
        v = np.median(self.img)
        print(v)
        lower = int(max(0, (1.0 - sigma) * v))
        print(lower)
        upper = int(min(255, (1.0 + sigma) * v))
        print(upper)
        edged = cv2.Canny(self.img, lower, upper)
        cv2.imshow("CANNY",edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        images = [self.img_blur, edged]
        plot_side_by_side(images, ["Original", "Canny Edge Sigma:0.33"], path, "Canny")
        return edged



def sobel_edge_detection(img_blur, path):
    """Input Image
        Output: list with img_sobelx, img_sobely, img_sobelxy , with edge detection"""
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    images = [img_blur,np.uint8(sobelx), np.uint8(sobely), np.uint8(sobelxy)]
    plot_side_by_side(images, ["Original", "Sobel in X", "Sobel in Y", "Sobel in XY"], path, "Sobel")
    return images

def laplacian_edge_detection(img, path):
    laplace = cv2.Laplacian(img,ddepth=cv2.CV_64F)
    images = [img,np.uint8(laplace)]
    plot_side_by_side(images, ["Original", "Laplace"], path, "Laplace")
    return np.uint8(laplace)
