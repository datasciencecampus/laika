import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"

from keras.optimizers import SGD
import h5py 

from model import model
from data import load_data


m = model()

print("Loading weights.")
m.load_weights("weights.hdf5")

print("Loading test data.")
images, labels = load_data(filtered="test-sample-filtered.txt", hdf5_data="data/testing.hdf5") 

#images, labels = images[:10], labels[:10] # dev

print("Compiling model.")
m.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False))

print("Evaluating model.")
loss, accuracy = m.evaluate(images, labels, verbose=1)
print("loss = {:.4f} accuracy = {:.4f}".format(loss, accuracy))

print("Predicting")
# (10, 65536, 7)
segments = m.predict(images)

print("Writing predictions.")
# store the or
hdf5 = h5py.File("predictions.hdf5", "w")
hdf5.create_dataset("predictions", data=segments)
hdf5.close()
