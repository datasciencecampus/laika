import json
import numpy as np
import pandas as pd
import cv2
import h5py
import os


def load_skynet_image(img_location, height, width):
    """load and pre-process satellite image.

    the result is a matrix of shape: (3, height, width), where 3 channels
    correspond to BGR order.

    the matrix is normalised within range [0, 1].
    """
    print("Processing image {}.".format(img_location))
    # satellite image 
    sample_img = cv2.imread(img_location)
    assert sample_img.shape == (height, width, 3)
        
    # image is in (height, width, channel) order.
    # our keras model expects (channel, height, width) order.
    # move channel axis to front.
    sample_img = np.rollaxis(sample_img, 2)
        
    # normalise the RGB values [0, 1]
    sample_img = sample_img/255

    return sample_img


def load_skynet_label(label_location, height, width, num_classes):
    """load and process the associated satellite image labels.

    the labels have been encoded as images, where pixel BGR values correspond to
    the pixel class:

    0x00, 0x00, 0x00 = background (unlabelled pixel)
    0x01, 0x01, 0x01 = class 1 (e.g., trees)
    ...., ...., ...,
    0x255, 0x255, 0x255 = class 255.

    result of this function is a matrix of vectors with shape shape: 
    (height*width, classes), where classes = total number of labelled classes.

    where m[:,0] = binary vector for class 0 (1 = pixel belongs to class, else 0)
          ...
          m[:, 255] = binary vector for class 255.
    """
    print("Processing label {}.".format(label_location))
    # labels encoded as an image.
    label_img = cv2.imread(label_location)
    assert label_img.shape == (height, width, 3)

    # 1 binary matrix per class. 
    masks = np.zeros((height, width, num_classes))
    for i in range(0, num_classes):
        masks[:,:,i] = 1*(label_img[:,:,0] == i)

    masks = masks.astype("float32")

    # flatten the labels -> vector.
    masks = np.reshape(masks, (height*width, num_classes))

    return masks


def load_skynet_data(skynet_data="../skynet-data/data", filtered="sample-filtered.txt", width=256, height=256):
    """"load images and image labels from skynet directory.
 
    result is a tuple of 2 matrices (images, labels) with 1 row per image.
    """
    # skynet images with _at least_ 1 labelled pixel.
    non_empty = pd.read_csv("{}/{}".format(skynet_data, filtered), sep=" ", header=None)

    ### images
    images = [load_skynet_image("{}/images/{}-{}-{}.jpg".format(skynet_data, *row[1:4]), height, width) for _, row in non_empty.iterrows()]

    ### labels
    # skynet class definitions
    with open(skynet_data + "/classes.json") as f:
        classes = json.load(f)

    # 1 extra class for "background" (unlabelled)
    num_classes = len(classes) + 1
    labels = [load_skynet_label("{}/labels/color/{}-{}-{}.png".format(skynet_data, *row[1:4]), height, width, num_classes) for _, row in non_empty.iterrows()]

    return np.array(images), np.array(labels)


def load_data(skynet_data="../skynet-data/data", filtered="sample-filtered.txt", hdf5_data="data/training.hdf5", width=256, height=256):
    """load training data.
   
    will load pre-processed data in data/training.hdf5 (if exists).
    else will do pre-processing, cache result and return.

    result is a tuple of 2 matrices (images, labels) with 1 row per image.
    """
    if not os.path.isfile(hdf5_data):
        print("Loading data from skynet.")
        images, labels = load_skynet_data(skynet_data, filtered, width, height)
        hdf5 = h5py.File(hdf5_data, "w")
        hdf5.create_dataset("images", data=images)
        hdf5.create_dataset("labels", data=labels)
        hdf5.close()
        return images, labels
    else:
        print("Loading data from HDF5.")
        hdf5 = h5py.File(hdf5_data, "r")
        images = hdf5.get("images")[()]
        labels = hdf5.get("labels")[()]
        hdf5.close()
        return images, labels 


if __name__ == "__main__":
    images, labels = load_data()
    print(images.shape)
    print(labels.shape)
