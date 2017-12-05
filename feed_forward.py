import os
import sys
import cv2 as cv
import numpy as np
from PIL import Image

from model import model
from visualise import class_heatmap, segment_image, combine_image

os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32, optimizer=fast_compile"


def segment(prediction):
    """assign class labels to pixels according to class probabilities.
    
    for each pixel, max(class_prob).
    
    returns an array. len = num pixels.
    """
    pixel_class = np.apply_along_axis(np.argmax, 1, prediction)
    return pixel_class.astype("uint8")


def load_image(img_src):
    """just load an image.

    returns numpy array.
    """
    return cv.imread(img_src)


def prep_image(img):
    """prepare image for input into network."""
    # move channel column to front. (RGB, height, width) 
    img = np.rollaxis(img, 2)
    # normalise pixels
    img = img/255
    return img


def load_model(weight_src="weights.hdf5"):
    """load pre-trained model"""
    m = model()
    m.load_weights(weight_src)
    return m


def write_class_heatmaps(prediction, output_dir="/tmp", height=256, width=256):
    """write class heatmaps.

    one image per class (for debugging)
    """
    magenta = np.array([
        [255,255,255],
        [255,247,243],
        [253,224,221],
        [252,197,192],
        [250,159,181],
        [247,104,161],
        [221,52,151],
        [174,1,126],
        [122,1,119],
        [73,0,106]])

    classes = prediction.shape[1]
    for i in range(0, classes):
        img = class_heatmap(magenta, prediction[:, i], 256, 256, True).resize((height, width))
        f = "{}/{:02d}_class_heatmap.png".format(output_dir, i)
        print("{} ...".format(f))
        img.save(f)

        
def write_segments(input_image, pixel_classes, output_dir="/tmp", height=256, width=256):
    """write segmented image

    output image is original with overlayed translucent segments."""
    class_colours = np.array([
        [255,255,255], # white: background/unlabeled
        [152,78,163],  # purple: residential
        [55,126,184],  # blue: commercial
        [255,255,51],  # yellow: industrial
        [77,175,74],   # green: vegetation
        [228,26,28],   # red: building
        [166,86,40]])  # brown: brownfield
    img = segment_image(class_colours, pixel_classes, height, width, bg_alpha=0, fg_alpha=192)
    combined_img = combine_image(img, input_image)
    f = "{}/segments.png".format(output_dir)
    
    combined_img.save(f)

    
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(sys.argv[0] + " <hdf5 weights> <img location> <output dir>")
        sys.exit(0)

    weight_src, img_src, output_dir = sys.argv[1:]

    # load input image
    input_image = load_image(img_src)
    img_arr = prep_image(input_image)
    assert img_arr.shape == (3, 256, 256)

    # load model
    m = load_model(weight_src)

    # get prediction   
    input_image = Image.fromarray(input_image)
    img_arr = np.reshape(img_arr, (1, 3, 256, 256))
    prediction = m.predict(img_arr)
    prediction = prediction[0]

    # class heatmaps
    write_class_heatmaps(prediction, output_dir)

    # segmented image
    pixel_classes = segment(prediction)
    write_segments(input_image, pixel_classes, output_dir)

    # summarise pixel assignments
    class_names = ["background", "residential", "commercial", "industrial", "vegetation", "building", "brownfield"]
    counts = np.asarray(np.unique(np.array(pixel_classes), return_counts=True)).T
    for class_n, count in counts:
        print("{} = {}".format(class_names[class_n], count))
