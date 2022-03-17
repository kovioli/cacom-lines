from multi_detection.config import imshape
import numpy as np
import cv2
import os
import json
from multi_detection.config import LOSS, backbone, model_name, EPOCHS, LEARNING_RATE, BATCH_SIZE

def preprocess_single(im, target_size = imshape[0]):
    vertical_cut = 250
    horizontal_cut = [80, 150]
    

    #image padding + resize:
    im = im[vertical_cut:-vertical_cut,horizontal_cut[0]:-horizontal_cut[1]]
    height, width, cc = im.shape
    side = max(height, width)

    ratio =  target_size / side
    
    black = (0, 0, 0)
    result = np.full((side, side, cc), black, dtype=np.uint8)

    xx = (side - width) // 2
    yy = (side - height) // 2

    result[yy:yy+height, xx:xx+width] = im
    result = cv2.resize(result, (target_size, target_size))

    shift = max(xx, yy)

    return result


def draw_prediction(im, pred, th=0.2, color=(30, 255, 30)):
    im = np.array(im, dtype = np.float64)#color.astype(np.float64)
    pred_upper = pred[:,:,0]
    pred_lower = pred[:,:,1]

    blank_upper = np.zeros(im.shape)
    blank_lower = np.zeros(im.shape)
    
    pred_upper[pred_upper<th] = 0
    pred_lower[pred_lower<th] = 0

    
    blank_upper[:,:,0] = pred_upper * color[0];
    blank_upper[:,:,1] = pred_upper * color[1];
    blank_upper[:,:,2] = pred_upper * color[2];
    
    blank_lower[:,:,0] = pred_lower * color[2];
    blank_lower[:,:,1] = pred_lower * color[0];
    blank_lower[:,:,2] = pred_lower * color[1];
    
    return (im + blank_upper + blank_lower)/255.0


def sorted_files(dir):
    """Sorts the files in a given directory

    Args:
        dir (str): directory path to be sorted

    Returns:
        list: list of sorted elements in a directory
    """
    #return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0].split('_')[1]))
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0].split('_')[1]))


def return_paths():
    dim_folder = "dim" + str(imshape[0])
    train_img_path = os.path.join('data', 'train_set', 'images')
    train_annot_path = os.path.join('data', 'train_set', 'annotations')

    val_img_path = os.path.join('data', 'validation_set', 'images')
    val_annot_path = os.path.join('data', 'validation_set', 'annotations')


    train_imgs = [os.path.join(train_img_path, x) for x in sorted(os.listdir(train_img_path))]
    train_annots = [os.path.join(train_annot_path, x) for x in sorted(os.listdir(train_annot_path))]

    val_imgs = [os.path.join(val_img_path, x) for x in sorted(os.listdir(val_img_path))]
    val_annots = [os.path.join(val_annot_path, x) for x in sorted(os.listdir(val_annot_path))]

    assert len(train_imgs) == len(train_annots) and len(val_imgs) == len(val_annots),\
    "The number of files in training or validation is different."

    return train_imgs, train_annots, val_imgs, val_annots


def create_config_json(train_dir, timestamp):  
    
    if isinstance(LOSS, str):
        loss_name = LOSS
    else:
        loss_name = LOSS.__name__
        
    info = {"model_name": model_name,
            "backbone": backbone,
            "freeze_backbone": False,
            "date": timestamp.split("_")[0],
            "time": timestamp.split("_")[1],
            "initial_lr": LEARNING_RATE,
            "loss_fn": loss_name,
            "image_size": imshape[0],
            "epoch": EPOCHS,
            "batch_size": BATCH_SIZE
            }

    filename = "config.json"
    with open(os.path.join(train_dir, filename), "w") as outfile:
        json.dump(info, outfile)

