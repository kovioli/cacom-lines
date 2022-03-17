import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import MeanIoU, Precision, PrecisionAtRecall, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pytz
import shutil
import datetime
from decimal import Decimal
from util import sorted_files, create_config_json, return_paths
from config import BATCH_SIZE, imshape, n_classes, ACTIVATION, LEARNING_RATE, LOSS, EPOCHS
from model import Unet
from data_generator import DataGenerator

"""
Function to fit the model on the preprocessed and split data 
"""

def dice(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


def dice_thresh(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    threshold_indices = y_pred>0.005
    y_pred[threshold_indices] = 0
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    

def iou(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


#get list of image and annot paths
train_imgs, train_annots, val_imgs, val_annots = return_paths()

#create generator for training data
#tg = DataGenerator(image_paths = train_imgs, annot_paths=train_annots, batch_size = BATCH_SIZE, augment = True)
tg = DataGenerator(image_paths = train_imgs, annot_paths=train_annots, batch_size = BATCH_SIZE, augment = True, filter = False)

#create generator for validation data
vg = DataGenerator(image_paths = val_imgs, annot_paths = val_annots, batch_size = BATCH_SIZE, augment = False, filter = False)

#create empty model
model = Unet(backbone_name="resnet50", input_shape=imshape, classes=n_classes, activation=ACTIVATION, encoder_freeze=False)


# CONTINUE TRAINING OPTION
#old_checkpoint_path = "./model.h5"
#model.load_weights(old_checkpoint_path)
# END CONTINUE TRAINING

model.compile(optimizer=Adam(LEARNING_RATE),
                loss=LOSS,
                metrics=[dice_loss, iou])
                #metrics=["categorical_crossentropy", iou])


timezone = pytz.timezone("Europe/Berlin")
timestamp = str(datetime.datetime.now(timezone).strftime("%Y-%m-%d_%H:%M:%S"))
os.mkdir('./models/' + timestamp)
training_dir = './models/' + timestamp
# checkpoint_path = os.path.join(training_dir, 'model', 'cp.ckpt')
checkpoint_path = os.path.join(training_dir, 'model.h5')
# checkpoint_dir = os.path.dirname(checkpoint_path)


#callback functions for the end of each epoch

tensorboard_callback = TensorBoard(os.path.join(training_dir, 'logs'), histogram_freq=1)

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, mode='min',
                             save_best_only=True, save_weights_only=True)

#checkpoint_abort = EarlyStopping(monitor = 'val_loss', patience = 22, verbose = 1)

#LR_scheduler = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 25,
#                                 verbose = 1, cooldown = 5, min_lr = 0.0000001)


callbacks = [
             tensorboard_callback,
             checkpoint,
             #checkpoint_abort,
             #LR_scheduler
             ]


create_config_json(training_dir, timestamp)

model_history = model.fit(x=tg,
                          validation_data = vg,
                          steps_per_epoch=len(tg),
                          epochs=EPOCHS,
                          verbose=1,
                          callbacks=callbacks
                          )