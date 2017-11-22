
# decile colours (white to magenta, purple)
# http://colorbrewer2.org/#type=sequential&scheme=RdPu&n=9
magenta_heat = [[0xff, 0xff, 0xff],
                [0xff, 0xf7, 0xf3],
                [0xfd, 0xe0, 0xdd],
                [0xfc, 0xc5 ,0xc0],
                [0xfa, 0x9f. 0xb5],
                [0xf7, 0x68, 0xa1],
                [0xdd, 0x34, 0x97],
                [0xae, 0x01, 0x7e],
                [0x7a, 0x01, 0x77],
                [0x49, 0x00, 0x6a]]

# class label colours
class_colours = [[0xff, 0xff, 0xff],
                 [0xe4, 0x1a, 0x1c], 
                 [0x37, 0x7e, 0xb8],
                 [0x4d, 0xaf, 0x4a],
                 [0x98, 0x4e, 0xa3],
                 [0xff, 0x7f, 0x00],
                 [0xff, 0xff, 0x33]]


def segment_image(pixel_classes):
    """create a segmented image from pixel class matrix.

    returns a PIL image object.
    """
    pass


def class_heatmap(probs):
    """create a heatmap from class probability matrix.

    returns a PIL iamge object.
    """
    pass 
