imshape = (640, 640, 3)

BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 3e-4
LOSS = "categorical_crossentropy"
# https://stats.stackexchange.com/questions/349096/cross-entropy-for-comparing-images
# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
# => dice gives a greater penalty for false positives (extra pixels that shouldn't have been recognised)
ACTIVATION = "softmax"
model_name = "unet"
backbone = "resnet50"
n_classes = 3