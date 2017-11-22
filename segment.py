import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"

import csv
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping

from model import model
from data import load_data

m = model()
m.load_weights("weights.hdf5")

images = None # load images to segment

segments = m.predict(images)


# for each image:
# 1) for each class, output a probability heathmap image.
# 2) output segment visualisation based on colours imported from json conf file.
# 2) output a matrix containing class probability deciles:
# 
#         10%, 20%, .., 100%
# class-1 x    x,   .., x
# class-2
# ...
# class-N
#
# where the sum of each row = 1: x = pixels in this decile / total pixels.
