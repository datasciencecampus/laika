# Notes


## Sentinel

### Uses for different wavelengths (nm)

src: [Classification and Segmentation of Satellite Orthoimagery Using Convolutional Neural Networks](https://ai2-s2-pdfs.s3.amazonaws.com/6269/c25f3cebe4dba83bd7feb78735796ffcdf6d.pdf)

* Red 630–690 Vegetation types, soils and urban features
* Green 510–580 Water, oil-spills, vegetation and man-made features
* Blue 450–510 Shadows, soil, vegetation and man-made features
* Yellow 585–625 Soils, sick foliage, hardwood, larch foliage
* Coastal 400–450 Shallow waters, aerosols, dust and smoke
* Seafloor 400–580 Synthetic image band (green, blue, coastal)
* NIR1 770–895 Plant health, shorelines, biomass, vegetation
* NIR2 860–1040 Similar to NIR1
* Pan sharpened 450–800 High-resolution pan and low-resolution multispectral
* Soil 585–625, 705–745, 770–895 Synthetic image band (NIR1, yellow, red edge)
* Land cover 400–450, 585–625, 860–1040 Synthetic image band (NIR2, yellow, coastal)
* Panchromatic 450–800 Blend of visible light into a grayscale
* Red edge 705–745 Vegetation changes
* Vegetation 450–510, 510–580, 770–895 Synthetic image band (NIR1, green, blue)

## Modelling

2 output types: 1) segmentation 2) object detection. Segmentation can be used 
for both land-use and object detection, pixels are classified individually. 
Final segments may need post-processing step (smoothing, removing "pepper".) 
Object detection is bounding box based or position of object centroid.

Current state of the art in image segmentation:

* U-Net
* SegNet (both segnet + u-net are encoder-decoder networks) (segnet outperformed by others by now) 
* DeepLab
* FCN (pioneering work. 2014. outperformed by others now)

Note, [PASCAL VOC Challenge performance evaluation](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6)
shows the current **state of the art** on VOC2012:

* DeepLab3-JFT, PSPNet...

### Progression rom coarse to fine inference

see [(2017) review on deep learning methods for semantic segmentation](https://arxiv.org/pdf/1704.06857.pdf)

1) **Classification**. Predicting which is the object in an image or even 
providing a ranked list if there are many of them.

2) **Localisation or detection**. Not only the classes but also additional
information regarding the spatial location of those classes, e.g., centroids or
bounding boxes.

3) **segmentation** dense predictions inferring labels for every pixel; this 
way, each pixel is labeled with the class of its en-closing object or region.

4) **instance segmentation** (separate labels for different instances of the 
same class) and even part-based segmenta-tion (low-level decomposition of 
already segmented classes into their components).

### Masks

0 for non existence of object/area, 1 if present. Note: If just using mask, it
will be difficult to differentiate between adjacent buildings. For this reason,
can also use a building outline mask. Then 3 labels: outside, edge, inside. As
per [details here](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53).


