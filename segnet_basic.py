# see:
# https://github.com/preddy5/segnet/blob/master/segnet.py
# https://github.com/0bserver07/Keras-SegNet-Basic/blob/master/SegNet-Basic.py
# https://github.com/imlab-uiip/keras-segnet/blob/master/build_model.py

from keras.models import Sequential
from keras.layers import Layer, Conv2D, BatchNormalization, Activation, MaxPooling2D, Reshape, Permute, UpSampling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import plot_model

from keras import backend as K
K.set_image_data_format("channels_first")


def model(img_channels=3, img_width=256, img_height=256):
    """define a basic segnet model."""
    model = Sequential()

    # encoder
    model.add(ZeroPadding2D(padding=1, input_shape=(img_channels, img_height, img_width)))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # decoder
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=512, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=2))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=256, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=2))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=128, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=2))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(filters=64, kernel_size=3, padding="valid"))
    model.add(BatchNormalization())

    # output layer
    model.add(Conv2D(filters=12, kernel_size=1, padding="valid"))
    model.add(Reshape((12, img_width*img_height), input_shape=(12, img_height, img_width)))
    model.add(Permute((2, 1)))
    model.add(Activation("softmax"))

    return model

#print(model.summary())
#plot_model(model, to_file="/tmp/x.png")
