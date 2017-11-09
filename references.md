y# References and resources

## Papers

### Satellite specific

#### 2017

* [Urban environments from satellite imagery - Github](https://github.com/adrianalbert/urban-environments) - Comparing urban environments using satellite imagery and convolutional neural nets. This is a very nice paper - lots of references to datasets. The focus is on open data. They use google maps api + [Urban Atlas](https://www.eea.europa.eu/data-and-maps/data/urban-atlas) for ground truth. Note: Urban Atlas last updated in 2010. They do not do pixel-wise classification, instead predict the classes inside the tile.

* [Deep Learning Based Large-Scale Automatic Satellite Crosswalk Classification](https://arxiv.org/pdf/1706.09302.pdf) - Zebra crossing identification using Google sat image data + Open street map zebra crossing locations. Associated code [here](https://github.com/rodrigoberriel/satellite-crosswalk-classification).

#### 2016

* [Classification and Segmentation of Satellite Orthoimagery Using Convolutional Neural Networks](https://ai2-s2-pdfs.s3.amazonaws.com/6269/c25f3cebe4dba83bd7feb78735796ffcdf6d.pdf) - Per pixel classification of vegetation, ground, roads, buildings and water. Per pixel is done in a sliding window: the pixel to be classified is the centre pixel of the window. The window has a number of dimensions (typically r, g, b) correspsonding to the wavelengths available from the satellite. Eg., may have an extra channel for near infrared. Standard CNN used: stacked conv-layers, fc layer and finally a softmax classifiier. ReLUs have been used in conv-layer. The fc layer used here is actually a 1000 hidden unit denoising auto-encoder. Has a post processing step: pixel-by-pixel classifications are likely to involve noise - e.g., a single pixel classified as an industrial area in the middle of a field/ "salt and pepper misclassifications" SLIC and other averaging has been used here. Interestingly, paper mentions using DBSCAN - a form of density clustering to solve this.  

**idea - Can Floyd–Steinberg or similar dithering algorithm be used in the convolution process?????**

* [Semantic Segmentation of Small Objects and Modeling of Uncertainty in Urban Remote Sensing Images Using Deep Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w19/papers/Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.pdf) - Uses 9cm ultra high res. data from ISPRS. They use the common F1 score to eval results.

* [First Experience with Sentinel-2 Data for Crop and Tree Species Classifications in Central Europe](http://www.mdpi.com/2072-4292/8/3/166/htm) - Random Forests. 

* [A Direct and Fast Methodology for Ship Recognition in Sentinel-2 Multispectral Imagery](http://www.mdpi.com/2072-4292/8/12/1033/htm)

* [Suitability of Sentinel-2 Data for Tree Species Classification in Central Europe](https://www.researchgate.net/profile/Markus_Immitzer/publication/303374163_Suitability_of_Sentinel-2_Data_for_Tree_Species_Classification_in_Central_Europe/links/573f0bbd08aea45ee844f238/Suitability-of-Sentinel-2-Data-for-Tree-Species-Classification-in-Central-Europe.pdf)

* [Benchmarking Deep Learning Frameworks for the Classification of Very High Resolution Satellite Multispectral Data](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/III-7/83/2016/isprs-annals-III-7-83-2016.pdf) - uses the [SAT-4 and SAT-6 airborne datasets](http://csc.lsu.edu/~saikat/deepsat/). 

* [Forecasting Vegetation Health at High Spatial Resolution - Github](https://github.com/JohnNay/forecastVeg) - Tool to produce short-term forecasts of vegetation health at high spatial resolution, using open source software and NASA satellite data that are global in coverage.

* [Automatic Building Extraction in Aerial Scenes Using Convolutional Networks](https://arxiv.org/pdf/1602.06564v1.pdf) - Describes a nice labelling technique: "Signed Distance Transform". +ve inside region, 0 at boundary and -ve outside. As described [here](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53).

#### 2015

* [Marker-Controlled Watershed-Based Segmentation of Multiresolution Remote Sensing Images](https://s3.amazonaws.com/academia.edu.documents/45383993/Marker-Controlled_Watershed-Based_Segmen20160505-26355-1xvg3iz.pdf)

* [Stable Mean-Shift Algorithm and Its Application to the Segmentation of Arbitrarily Large Remote Sensing Images](http://faculty.wwu.edu/wallin/envr442/pdf_files/Michel_etal_2015_Stable_mean_shift_algorithm_app_to_segmentation_large_remote_sensing_images_IEEE_Trans_Geosci_RS.pdf)

* [Satellite Image Segmentation based on YCbCr Color Space](http://www.indjst.org/index.php/indjst/article/viewFile/51281/46256)

* [Unsupervised Deep Feature Extraction for Remote Sensing Image Classification](https://arxiv.org/pdf/1511.08131.pdf)

#### 2014

* [Remote Sensing Image Segmentation by Combining Spectral and Texture Features](http://web.cse.ohio-state.edu/~wang.77/papers/YWL.tgrs14.pdf)

#### 2013

* [Geographic Object-Based Image Analysis – Towards a new paradigm](http://www.sciencedirect.com/science/article/pii/S0924271613002220)

* [Hyperspectral Remote Sensing Data Analysis and Future Challenges](https://pdfs.semanticscholar.org/4c97/98a76945dc2651feebfe0a2e35ae31abe64f.pdf)

### Modelling specific

#### 2017

* [review on deep learning methods for semantic segmentation](https://arxiv.org/pdf/1704.06857.pdf)

* [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)

* [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf). Bleeding edge.

### 2016

* [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)

#### 2015

* [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) - \*\*\*

## Models

### Satellite

* [Github - DeepOSM](https://github.com/trailbehind/DeepOSM) - \*\*\* Train a deep learning net with OpenStreetMap features and satellite imagery.. Data is combo of osm classifications + [NAIP data](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (National Agriculture Imagery Program) / us specific. TensorFlow. Predict if the center 9px of a 64px tile contains road.

* [Github - Deep networks for Earth Observation](https://github.com/nshaud/DeepNetsForEO) - Pretrained [Caffe](https://github.com/bvlc/caffe) [SegNet](https://arxiv.org/abs/1511.02680) models trained on [ISPRS Vaihingen](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset and [ISPRS Potsdam](http://www2.isprs.org/potsdam-2d-semantic-labeling.html) datasets. Uses OSM tiles. SegNet = all pixels. Note, there is also a probabilistic extension to segnet. SegNet has also been used with GSV imagery.. 

* [Github - Raster Vision](https://github.com/azavea/raster-vision) - Deep learning for aerial/satellite imagery. [ResNet50](https://arxiv.org/abs/1512.03385), [Inception v3](https://arxiv.org/abs/1512.00567), [DenseNet121](https://arxiv.org/abs/1608.06993), [DenseNet169](https://arxiv.org/abs/1608.06993) models. Uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

* [Github - Deep Learning Tutorial for Kaggle Ultrasound Nerve Segmentation competition, using Keras](https://github.com/jocicmarko/ultrasound-nerve-segmentation) - This monster was used in 1tst place solution to DSTL Kaggle satellite challenge. (U-net)

* [Predicting Poverty and Developmental Statistics from Satellite Images using Multi-task Deep Learning - Github](https://github.com/agarwalt/satimage) - Keras, Google sat. images, India cencus data.

* [Using convolutional neural networks to pre-classify images for the humanitarian openstreetmap team (HOT & mapgive)](https://github.com/larsroemheld/OSM-HOT-ConvNet) - Donated MapGive data + Openstreetmap. Caffe, SegNet.

* [OSMDeepOD - OSM and Deep Learning based Object Detection from Aerial Imagery](https://github.com/geometalab/OSMDeepOD) - Object detection from aerial imagery using open data from OpenStreetMap. - Not sure where images come from. TensorFlow, pretrained Inception V3.

* [ssai-cnn - Semantic Segmentation for Aerial / Satellite Images with Convolutional Neural Networks](https://github.com/mitmul/ssai-cnn) - uses [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/). Implementation of CNN, based on methods in this [paper](http://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003)

* [SpaceNetChallenge models - Github](https://github.com/SpaceNetChallenge/BuildingDetectors) - Winning solutions to spacenet building detection challenge.

1) [random forests](https://github.com/SpaceNetChallenge/BuildingDetectors/tree/master/wleite) Java :)

2) [Multi-scale context aggregation by dilated convolutions](https://github.com/SpaceNetChallenge/BuildingDetectors/tree/master/marek.cygan) (Tensorflow)

3) [Instance-aware semantic segmentation via multi- task network cascades](https://github.com/SpaceNetChallenge/BuildingDetectors/tree/master/qinhaifang) (Caffe)

4) [Fully Convolutional Network (FCN) + Gradient Boosting Decision Tree (GBDT)](https://github.com/SpaceNetChallenge/BuildingDetectors/tree/master/fugusuki) (Keras)

5) [FCN](https://github.com/SpaceNetChallenge/BuildingDetectors/tree/master/bic-user) (Keras)


### General

* [Awesome Semantic Segmentation - Github](https://github.com/mrgloom/awesome-semantic-segmentation)

### SegNet

Keras implementations:

(note these are all the basic version from the paper.

* [keras-segnet](https://github.com/imlab-uiip/keras-segnet) 65 stars.
* [SegNet-Basic])https://github.com/0bserver07/Keras-SegNet-Basic) 30 stars
* [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras) 10 stars. Also some other's (u-net, fcn etc.)
* [keras-SegNet](https://github.com/ykamikawa/keras-SegNet) 1 star (from me :p).

### U-Net

Keras implementations:

* [ultrasound-nerve-segmentation](https://github.com/jocicmarko/ultrasound-nerve-segmentation) 516 stars. From Kaggle.
* [another ultrasound-nerve-segmentation](https://github.com/EdwardTyantov/ultrasound-nerve-segmentation) 100 stars. Kaggle.
* [u-net](https://github.com/yihui-he/u-net) 90 stars.
* [ZF\_UNET\_224\_Pretrained\_Model](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) 61 stars. \*\*\* used for 2nd place in DSTL challenge.
* [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras) 10 stars. Same as above.
* [mobile-semantic-segmentation](https://github.com/akirasosa/mobile-semantic-segmentation) 6 stars.

### DeepLab

Note, there seems to be no Keras implementation.
Tensorflow implementations:

* [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) 480 stars.

### Dilated Convolutions

* [segmentation\_keras](https://github.com/nicolov/segmentation_keras) 153 stars.

### PSPNet (Pyramid Scene Parsing Network)

Bleeding edge. More details [here](https://hszhao.github.io/projects/pspnet/index.html)

* [PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow) 67 stars.

### DSTL - Kaggle

1. [U-Net](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/)

2. not known 

3. [Another U-Net](http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/)

4. [modified U-Net](https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)

5. [pixel-wise logistic regression model](https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly)

### Understaning Amazon from space - Kaggle

1. [11 convolutional neural networks](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/) - This is pretty nice, details winners' end-to-end architecture.


## Datasets

### Raw data

Sentinel 2 data looking the most promising. (10m resolution for R,G,B + NIF) 
(20m for shorter wavelengths). - see [MultiSpectral Instrument (MSI) Overview](https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument) for instrument details.

**idea -- downscale higher resolution pre-labelled data to 10m per pixel if can not find pre-labelled 10m dataset**


* [Copernicus Open Access Hub](https://scihub.copernicus.eu/) - The Copernicus Open Access Hub (previously known as Sentinels Scientific Data Hub) provides complete, free and open access to [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1), [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) and [Sentinel-3](https://sentinel.esa.int/web/sentinel/missions/sentinel-3).

* [Satellite Applications Catapult - Data discovery hub](http://data.satapps.org/) - Very nice. Lists pretty much everything.

* [Satellite Applications Catapult - Sentinel Data Access Service (SEDAS)](https://sa.catapult.org.uk/facilities/sedas/]) - (API) portal enabling end-users to search and download data from Sentinel 1 & 2 satellites. It aims to lower barriers to entry and create the foundations for an integrated approach to satellite data access.

* [Google](https://developers.google.com/maps/documentation/static-maps/) - 25k per day... :) - would also be easy to build testing tool from this..  

### Pre-labelled

* [DSTL - Kaggle](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

* [Draper Satellite Image Chronology - Kaggle](https://www.kaggle.com/c/draper-satellite-image-chronology)

* [Understanding the Amazon from Space - Kaggle](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

* [Ships in Satellite Imagery - Kaggle](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) - From [Open California](https://www.planet.com/products/open-california/) dataset.

* [2D Semantic Labeling - Vaihingen data](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) - This is from a true orthophotographic survey (from a plane). The data is 9cm resolution!

* [SAT-4 and SAT-6 airborne datasets](http://csc.lsu.edu/~saikat/deepsat/) - very high res (1m) whole of US. 65 Terabytes. (DeepSat) 1m/px. (2015). SAT-4: barren land, trees, grassland, other. SAT-6: barren land, trees, grassland, roads, buildings, water bodies. (RGB + NIR) 


#### Land-use

* [Skynet - Data pipeline for machine learning with OpenStreetMap - Github](https://github.com/developmentseed/skynet-data) - Uses OSM QA tiles.

* [UC Merced Land Use Dataset](http://vision.ucmerced.edu/datasets/landuse.html) - 2100 256 × 256, 1m/px aerial RGB images over 21 land use classes. US specific. (2010)

* [Open street map landuse](http://osmlanduse.org/#11/-3.23866/51.57133/0/) - OSM landuse visualised in this tool. Some studies have used this in combo. with google sat. images.

## Other

* [Terrapattern](http://www.terrapattern.com/about) - alpha version of Terrapattern: "similar-image search" for satellite photos. They use a ResNet. This is truely awesome.

* [European Environment Agency - Urban Atlas](https://www.eea.europa.eu/data-and-maps/data/urban-atlas) - Land cover data for Large Urban Zones with more than 100.000 inhabitants.

* [Satellite imagery - what can hospital carparks tell us about disease outbreak?](http://online.qmags.com/CMG0414/Default.aspx?pg=97&mode=2&imm_mid=0bb43a&cmp=em-strata-na-na-newsltr_20140423_elist#pg97&mode2)

* [TEP Urban platform - Thematic Apps](https://urban-tep.eo.esa.int/#!thematic) - Lots of things. [Urban density](https://urban-tep.eo.esa.int/geobrowser/?id=guf#!&context=GUFDensity%2FGUF-DenS2012) / GUF. Derived from sat data.

* [SpaceNetChallenge utlitiles - Github](https://github.com/SpaceNetChallenge/utilities) - Packages intended to assist in the preprocessing of SpaceNet satellite imagery data corpus to a format that is consumable by machine learning algorithms.

## Tools/utilities

* [Sentinelsat - Github](https://github.com/sentinelsat/sentinelsat) - Utility to search and download Copernicus Sentinel satellite images.

## Visualisations/existing tools

* [Global Forest Watch: An online, global, near-real time forest monitoring tool - Github](https://github.com/Vizzuality/gfw) - Really nice site. Also has an option to overlay Sentinel 2 images from specific dates.

* [Visualize AWS Sentinel-2 data in different band combinations](http://apps.sentinel-hub.com/sentinel-playground/?source=S2&lat=51.532348250305354&lng=-3.2268905639648438&zoom=12&preset=5_VEGETATION_INDEX&layers=B04,B03,B02&maxcc=20&gain=1&gamma=1&time=2015-01-01%7C2017-10-31&atmFilter=&showDates=false)

## Meta

* [Github satellite-imagery view](https://github.com/topics/satellite-imagery) - Good starting point.

* [Awesome Sentinel - Github](https://github.com/Fernerkundung/awesome-sentinel) - Sat data tools/utils/visualisations etc.

## Projects using Sentinel data

* [sen2agri](http://www.esa-sen2agri.org/) - Preparing Sentinel-2 exploitation for agriculture monitoring. 2017 paper/demo: [Production of a national crop type map over the Czech Republic using Sentinel-1/2 images](http://www.esa-sen2agri.org/wp-content/uploads/docs/CzechAgri%20Final%20Report%201.2.pdf)

## Platforms

* [Earth on AWS Build planetary-scale applications in the cloud with open geospatial data](https://aws.amazon.com/earth/) - Earth Observation data on AWS.


## Future competitions

* [Topcoder SpaceNet challenge round 3](https://crowdsourcing.topcoder.com/spacenet) - release: 11/13/17, start: 11/18/17, end: 12/1/17.

## Blogs

* [The DownLinQ](https://medium.com/the-downlinq) - Nice blog from CosmiQ.

* [2016: A review of deep learning models for semantic segmentation](http://nicolovaligi.com/deep-learning-models-semantic-segmentation.html)

* [2017: http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review) 

