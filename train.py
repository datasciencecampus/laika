# train image segmentation model.

import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"

import csv
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping

from model import model
from data import load_data

epochs = 2

m = model()
print("Loaded model.")
print(m.summary())
#plot_model(model, to_file="/tmp/x.png")

opt = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
m.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("Model compiled.")

images, labels = load_data()
images, labels = images[:100], labels[:100] # dev

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

# callbacks broken in latest keras version.
#call = []
#call.append(CSVLogger("train.log", separator=",", append=False))
#call.append(EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1, mode="min"))
hist = m.fit(x=images, y=labels, validation_data=(images_val, labels_val), batch_size=16, epochs=epochs, verbose=1)
print("Training complete.")

print("Saving weights.")
m.save_weights("weights.hdf5")

print(hist.history)

print("Saving training history.")
# {"a": [1,2,3], "b": [4,5,6]} ->
# [[1,4], [2,5], [3,6]]
hist_dict = hist.history
headers = ["loss", "val_loss", "val_acc", "acc"]
train_hist = [list(x) for x in zip(*[hist_dict[k] for k in headers])]

with open("training_log.csv", "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(headers)
    csv_writer.writerows(train_hist)
