import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"

import numpy as np

from model import model
from data import load_data

m = model()

print("Loading weights.")
m.load_weights("weights.hdf5")

print("Loading test data.")
images, labels = load_data(filtered="test-sample-filtered.txt", hdf5_data="data/testing.hdf5") 

images, labels = images[:10], labels[:10] # dev

print("Predicting")
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
