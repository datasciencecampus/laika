import numpy as np
from PIL import Image


def combine_image(label_img, sat_img, alpha=128):
    """
    combine 2 images.
    """
    assert label_img.mode == "RGBA"
    combined_img = sat_img.copy()
    combined_img.paste(label_img, (0, 0), label_img)
    return combined_img


def class_heatmap(heat_deciles, pixel_probabilities, height, width, invert=False, ignore_cut_off=-1, bg_alpha=0, fg_alpha=255):
    bins = np.floor(pixel_probabilities*10) # discretise
    bins = bins.astype("uint8")
    heat = heat_deciles[::-1] if invert else heat_deciles
    img_arr = np.reshape(heat[bins], (height, width, 3))
    img_arr = img_arr.astype("uint8")
    img = Image.fromarray(img_arr)
    img = img.convert("RGBA")
    # ignore alpha
    img_arr = np.array(img)
    cut_out = np.reshape(bins, (height, width)) <= ignore_cut_off
    img_arr[cut_out, 3] = bg_alpha
    # fg alpha
    img_arr[np.logical_not(cut_out), 3] = fg_alpha
    img = Image.fromarray(img_arr)
    return img


def segment_image(class_colours, pixel_classes, height, width, bg_alpha=0, fg_alpha=255):
    """visualise pixel classes"""
    segment_colours = np.reshape(class_colours[pixel_classes], (height, width, 3))
    segment_colours = segment_colours.astype("uint8")
    img = Image.fromarray(segment_colours)
    # set backgroud/unlabeled pixel alpha to 0.
    # note to self: do with numpy
    img = img.convert("RGBA")
    arr = np.array(img)
    arr = np.reshape(arr, (height*width, 4))
    background = np.where(pixel_classes == 0)
    arr[background, 3] = bg_alpha
    background = np.where(pixel_classes > 0)
    arr[background, 3] = fg_alpha
    arr = np.reshape(arr, (height, width, 4))
    return Image.fromarray(arr)
