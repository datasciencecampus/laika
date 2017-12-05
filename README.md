# Project Laika

> Experiments with satellite image data.


## Synopsis

The goal of *project laika* is to research potential sources of satellite image
data and to implement various algorithms for satellite image classification, 
object detection and segmentation. Laika is a precursor for a production ready,
*end-to-end* satellite image processing framework and also is intended to serve
as a protyping environment for testing recent and future developments.


## Goals

1. Review potential sources of satellite imagery
2. Implement an existing satellite segmentation algorithm
3. Document and communicate the algorithm
4. Define the high level pipeline
5. Develop and test a model for collaborative project work

Please consult the [success criteria](success_criteria.md) for a detailed 
breakdown of this project's goals.


## Satellite themes of interest

In general, satellite image processing themes can be categorised into two main
themes:


### Earth Observation (EO)

The field of [Earth observation](https://en.wikipedia.org/wiki/Earth_observation)
is concerned with monitoring the status of the planet with various *sensors*, 
which includes, but is not limited to, the use of satellite data for monitoring
large areas of the earth's surface at regular, frequent intervals.

EO is a broad area, which may cover water management, forestry, agriculture,
urban fabric and land-use/cover in general.

A good example of EO is the use of the [normalised difference vegetation index (NDVI)](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)
for monitoring nationwide vegetation levels. 

This sub-field does not depend on very high resolution data since it is mainly
concerned with quantifying some aspect of very large areas of land.


### Object detection

In contrast to EO, the field of object detection from satellite data is 
principally concerned with the localisation of specific objects as opposed to 
general land-use cover. 

A good example of Object detection from satellite data is 
[counting cars in carparks](https://medium.com/the-downlinq/car-localization-and-counting-with-overhead-imagery-an-interactive-exploration-9d5a029a596b)
from which various indicators can be derived, depending on the context. For 
example, it would be possible to derive some form of consumer/retail indicator
by periodically counting cars in super market car parks.

This sub-field will depend on very high resolution data depending on the 
application.


## Computer vision themes of interest

In addition to the two main satellite image processing themes of interest (EO
and object detection), there are four more general image processing sub-fields
which may be applicable to a problem domain. From "easiest" to most difficult:

1. **Classification**. At the lowest level, the task is to identify which 
objects are present in a scene. If there are many objects, the output may be an
ordered list sorted by amount or likelyhood of the object being present in the
scene. The classification may also extend beyond objects to abstract concepts
such as aesthetic value.

2. **Detection**. The next level involves localisation of the entities/concepts
in the scene. Typically this will include a bounding-box around identified 
objects, and/or object centroids.

3. **Segmentation**. This level extends classification and detection to include
pixel-by-pixel class labeling. Each pixel in the scene must be labeled with a
particular class, such that the entire scene can be described. Segmentation is
particularly appropriate for land-use cover. In addition, segmentaiton may be
extended to provide a form of augmented bounding-box: pixels outside of the 
bounding box area can be negatively weighted, pixels on the border +1 and pixels
inside the region [0, 1] inversely proportionate to the distance form the 
bounding box perimeter.

4. **Instance segmentation**. Perhaps the most challenging theme: In addition to
pixel-by-pixel segmentation, provide a segmented object hierarchy, such that 
objects/areas belonging to the same class may be individually segmented. E.g.,
segments for cars *and* car models. Commercial area, office within a commercial
area, roof-top belonging to a shop etc. 

The initial version of this project focuses on **option 3**: image segmentation
in the domain of **both** Earth Observation and object detection.


## Data sources

There are three types of data of interest for this project.

1. **Raw image data**. There are numerous sources for satellite image data, 
ranging from lower resolution (open) data most suited for EO applications, 
through to high resolution (mostly propitiatory) data-sets. 

2. **Pre-labeled image data**. For training an image classificiation, object
detection or image segmentation supervised learning model, it is necessary to
obtain ample training instances along with associated ground truth labels. In
the domain of general image classification, there exist plenty of datasets which
are mainly used to benchmark various algorithms.

3. **Image labels**. It will later be required to create training datasets with
arbitrary labeled instances. For this reason, a source of ground-truth and/or a
set of tools to facilitate image labeling and manual image segmentation will be
necessary.


### Raw image data 

This project (to date) focuses specifically on **open data**. The 2 main data
sources for EO grade images come from the 
[Sentinel-2](https://en.wikipedia.org/wiki/Sentinel-2) and
[Landsat-8](https://en.wikipedia.org/wiki/Landsat_8) satellites. Both satellites
host a payload of multi-spectrum imaging equipment.


#### Sentinel 2 (ESA)

The Sentinel-2 satellite is capable of sensing the following wavelengths:

| Band                     | Wavelength (&mu;m) | Resolution (m) |
| ------------------------ | ------------------ | -------------- |
| 01 – Coastal aerosol     | 0.443              | 60             |
| 02 – Blue                | 0.490              | 10             |
| 03 – Green	           | 0.560              | 10             |
| 04 – Red                 | 0.665              | 10             |
| 05 – Vegetation Red Edge | 0.705              | 20             |
| 06 – Vegetation Red Edge | 0.740              | 20             |
| 07 – Vegetation Red Edge | 0.783              | 20             |
| 08 – NIR                 | 0.842              | 10             |
| 8A – Narrow NIR	   | 0.865              | 20             |
| 09 – Water vapour        | 0.945              | 60             |
| 10 – SWIR – Cirrus       | 1.375              | 60             |
| 11 – SWIR                | 1.610              | 20             |
| 12 – SWIR                | 2.190              | 20             |

!["Sentinel 2"](img/sentinel_2.jpg)

The visible spectrum captured by Sentinel-2 is the highest (open) data 
resolution available: 10 metres per pixel. Observations are frequent: Every 5
days for the same viewing angle.

* [sentinel-playground](http://apps.sentinel-hub.com/sentinel-playground/?source=S2&lat=51.653814904471545&lng=-3.021240234375&zoom=8&preset=2_COLOR_INFRARED__VEGETATION_&layers=B04,B03,B02&maxcc=20&gain=1.0&gamma=1.0&time=2015-01-01%7C2017-12-05&atmFilter=&showDates=false) - This is a nice demo showing the available sentinel-2 bands.

* [Copernicus Open Access Hub](https://scihub.copernicus.eu/) - The Copernicus Open Access Hub (previously known as Sentinels Scientific Data Hub) provides complete, free and open access to [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1), [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) and [Sentinel-3](https://sentinel.esa.int/web/sentinel/missions/sentinel-3).

* [Satellite Applications Catapult - Data discovery hub](http://data.satapps.org/) - Very nice. Lists pretty much everything.

* [Satellite Applications Catapult - Sentinel Data Access Service (SEDAS)](https://sa.catapult.org.uk/facilities/sedas/]) - (API) portal enabling end-users to search and download data from Sentinel 1 & 2 satellites. It aims to lower barriers to entry and create the foundations for an integrated approach to satellite data access.

* [Google](https://developers.google.com/maps/documentation/static-maps/) - 25k per day... :) - would also be easy to build testing tool from this.. Note that this data is &lt; 1m resolution, so perfectly suitable for object detection.

* [Earth on AWS](https://aws.amazon.com/earth/) - Lots of data sources (inc. sentinel and landsat) + platform for large-scale processing. 


#### Landsat-8 (NASA)

The Landsat-8 satellite is limited to 30m resolution accross all wavelengths
with the exception of it's panchromatic sensor, which is capable of capturing
15m per pixel data. The revisit frequency for landsat is 16 days.

| Band                           | Wavelength (&mu;m) | Resolution (m) |
| ------------------------------ | ------------------ | -------------- |
| 01 - Coastal / Aerosol         | 0.433 – 0.453      |	30             |
| 02 - Blue                      | 0.450 – 0.515      | 30             |
| 03 - Green                     | 0.525 – 0.600      | 30             |
| 04 - Red                       | 0.630 – 0.680      | 30             |
| 05 - Near Infrared             | 0.845 – 0.885      | 30             |
| 06 - Short Wavelength Infrared | 1.560 – 1.660      | 30             |
| 07 - Short Wavelength Infrared | 2.100 – 2.300      | 30             |
| 08 - Panchromatic              | 0.500 – 0.680      | 15             |
| 09 - Cirrus                    | 1.360 – 1.390      | 30             |


!["Landsat 8"](img/landsat_8.png)


### Pre-labeled image data

There are numerous sources of pre-labeled image data available. Recently, there
have been a number of satellite image related competitions hosted on Kaggle and
TopCoder. This data may be useful to augment an existing dataset, to pre-train
models or to train a model for later use in an ensemble.

* [DSTL - Kaggle](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

* [Draper Satellite Image Chronology - Kaggle](https://www.kaggle.com/c/draper-satellite-image-chronology)

* [Understanding the Amazon from Space - Kaggle](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

* [Ships in Satellite Imagery - Kaggle](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) - From [Open California](https://www.planet.com/products/open-california/) dataset.

* [2D Semantic Labeling - Vaihingen data](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) - This is from a true orthophotographic survey (from a plane). The data is 9cm resolution!

* [SAT-4 and SAT-6 airborne datasets](http://csc.lsu.edu/~saikat/deepsat/) - Very high res (1m) whole of US. 65 Terabytes. (DeepSat) 1m/px. (2015). SAT-4: barren land, trees, grassland, other. SAT-6: barren land, trees, grassland, roads, buildings, water bodies. (RGB + NIR) 


### Object and land-use labels

There exist a number of land-use and land-cover datasets. The main issue is 
dataset age: If the objective is to identify construction sites or urban sprawl
then a dataset &gt; 1 year is next to useless, unless it can be matched to 
imagery from the same time period which would then only be useful for the 
creation of a training dataset.

The most promising source (imo) is the [OpenStreetMap](https://www.openstreetmap.org/)
project since it is updated constantly and contains an **extensive** hierarchy
of relational objects. There is also the possibility to contribute to the OMS 
project should manual labeling be necessary.

* [UC Merced Land Use Dataset](http://vision.ucmerced.edu/datasets/landuse.html) - 2100 256x256, 1m/px aerial RGB images over 21 land use classes. US specific. (2010)

* [Open street map landuse](http://osmlanduse.org/#11/-3.23866/51.57133/0/) - OSM landuse visualised in this tool. Some studies have used this in combo. with google sat. images.

* [European Environment Agency - Urban Atlas](https://www.eea.europa.eu/data-and-maps/data/urban-atlas) - Land cover data for Large Urban Zones with more than 100.000 inhabitants.


## Current competitions

* [13/11/17 - 28/02/18 SpaceNet challenge round 3](https://crowdsourcing.topcoder.com/spacenet) - TopCoder. Extract navigable road networks that represent roads from satellite images.

