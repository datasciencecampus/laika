import os

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cpu, floatX=float32, optimizer=fast_compile"

from keras.optimizers import SGD

from model import model
from data import load_data

m = model()
print("Loaded model.")
print(m.summary())
#plot_model(model, to_file="/tmp/x.png")

opt = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
m.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
print("Model compiled.")

images, labels = load_data()
print("Loaded data.")

hist = m.fit(x=images, y=labels, batch_size=14, epochs=100, verbose=1)
print("Training complete.")
