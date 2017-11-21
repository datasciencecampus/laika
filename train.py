import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"

import numpy as np
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

from model import model
from data import load_data

epochs = 1

m = model()
print("Loaded model.")
print(m.summary())
#plot_model(model, to_file="/tmp/x.png")

opt = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
m.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("Model compiled.")

images, labels = load_data()

# shuffle dataset.
total = len(images)
p = np.random.permutation(total)
images, labels = images[p], labels[p]

# 80/20 split.
holdback = int(0.2*total)
images_val, labels_val = images[-holdback:], labels[-holdback:]
images, labels = images[:-holdback], labels[:-holdback]

print("Loaded {} instances.".format(total))
print("Training = {}, Validation = {}".format(len(images), len(images_val)))

callbacks = []
callbacks.append(CSVLogger("train.log", separator=",", append=False))
callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="min"))
hist = m.fit(x=images, y=labels, validation_data=(images_val, labels_val), batch_size=16, epochs=epochs, verbose=1, callbacks=[call])
print("Training complete.")

print(hist)
