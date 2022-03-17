from multi_detection.model import Unet
from multi_detection.config import backbone, imshape, n_classes, ACTIVATION
from multi_detection.util import preprocess_single, draw_prediction
import cv2
import numpy as np
from multi_detection.helper_curvefitting import curve_fit
import json
import os


def predict(im, model_path=r'..\multi_detection\model.h5'):
    """
    Performs prediction on an input image, and gives back the prediction drawn on top, together with the coordinates of the fitted lines

    input:
        im [numpy array]:
            image on which prediction is performed
        model_path [str]:
            path for the pretrained model weights (in .h5 format)
    
    output:
        info [dict]:
            dictionary that contains the points of the fitted curve
        drawing [numpy array]:
            image with the predicted lines drawn on top (size: (640, 640, 3))
    """
    #print(os.getcwd())

    model = Unet(backbone_name=backbone, input_shape = imshape, classes = n_classes, activation = ACTIVATION, encoder_weights = None, weights = model_path)
    
    #preprocess image (to fit the model input shape, (640x640x3)) - padding + cutting
    im_reshape = preprocess_single(im)

    # perform prediction on image
    pred = model.predict(np.expand_dims(im_reshape, axis = 0)).squeeze()
    #cv2.imshow("asdfasdf", pred[:,:,1]+pred[:,:,0])

    info = curve_fit("test", pred)

    drawing = draw_prediction(im_reshape, pred, th=0.2)

    return drawing, info

"""read in image and perform prediction"""
#im_orig_path = "./data/test_set/images/L9GE0MHG.png"
#im = cv2.imread(im_orig_path, 1)



#drawing, info = predict(im)

"""show image"""
#cv2.imshow("prediction", drawing)
#cv2.waitKey(0)

"""save data points"""
#with open(f'filename.json', 'w') as fp:
#    json.dump(info, fp)