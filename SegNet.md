# SegNet

> A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling

From SegNet research [website](http://mi.eng.cam.ac.uk/projects/segnet/):

"SegNet is a deep encoder-decoder architecture for multi-class pixelwise segmentation researched and developed by members of the Computer Vision and Robotics Group at the University of Cambridge, UK."

**Paper**: [Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017](http://arxiv.org/abs/1511.00561)

## Existing implementations

* [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet) - Original Caffe implementation.
* [imlab-uiip/keras-segnet](https://github.com/imlab-uiip/keras-segnet) - Keras implementation (all 13 layers)
* [0bserver07/Keras-SegNet-Basic](https://github.com/0bserver07/Keras-SegNet-Basic) - Keras implementation (4 layer version)

## Notes

* Encoder/Decoder architecture.
* Encoder layer is topologically the same as the convolutional layers in a [VGG16](https://arxiv.org/abs/1409.1556) network.
* Fully connected layers in VG16 have been removed.
* Decoder layer mirrors the encoder layer: One convultional layer for each layer in the encoder network.
* Problem with using a VGG16 network for **pixel wise** labelling is that it 
requires different stages of training: 1) train network to be able to 
**classify** the presence of objects in the scene. 2) train a network to output
pixel labels or produce the labels with a **heatmap** approach. Also, having 
fully connected layers = more parameters.

## Deps.

* theano
* keras
* pydot
* graphviz


```
pip3 install pydot-ng
brew install install graphviz
```
