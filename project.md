# Project outline

(for the "working with the ons data science campus: what we need to know about
the project" document.

## About the project

### What is the project's *primary* objective?

The Data Science Campus have many projects in their backlog that involve some 
form of computer vision challenge. These range from using satellite imagery for
land use problems to using crowd sourced photographs for object detection.

Project Laika aims to develop an satellite image processing pipeline that can be
used to satisfy the computer vision component of future projects involving the
use of satellite imagery.

### Summary of the project, including background

The ultimate aim of this project is to devliver a generic satellite imagery 
processing pipeline.

This is a large project, and as such will be delivered in a series of phases.

The first phase will involve solving a specific satellite imagery 
classification/segmentation problem. As a starting point, we plan to be able to
differentiate between vegetation and non-vegetation using an open satellite 
imagery dataset. Solving this phase will involve a number of sub-tasks, 
including, but not limited to:

* Background research (existing techniques)
* Data provision, including the sourcing of pre-labelled images.
* Exploratory data analysis: Apply some non-ML techniques and conduct feature
engineering.
* Model development (which will most likely involve convolutional neural 
networks)
* Prototype/proof of concept development: an initial end-to-end pipeline that
takes as input a batch of images and outputs object labels (in the form of 
segmented polygons?)
* Visualisation: A lightweight tool to visualise results.

**We aim to complete this phase within 2 weeks.**

The first phase is mainly a reconnaissance/internal knowledge building exercise.
The primary objective is to create an end-to-end prototype from which we can
extract **generic** features for use in a more complete system. For example, in
solving a specific segmentation problem, we may notice the need for a generic
"model aggregation" layer, should combining models be an important part of the 
task.

### Outcomes: How will this benefit the following groups:

* __General public__: There is strong scope to produce work in this area which
would be of benefit the the general public. Specifically, this project will lay
the foundation for future projects which have a strong intersection with SDGs.

* __Government__: Potential collaborations with env. accounts, defra etc. Lots
of scope for use of data in policy making, urban planning etc. Scope to provide
econonomic insights.

* __Industry__: Use of satellite data in "alternative data" is a current trend.

* __International community__: SDGs (as mentioned previously). Also, work would
be generic, not limited to the UK. We hope to produce mainly open code.

* __Another other groups?__: -

### Outputs: What are the key deliverables from the project?

* end-to-end processing pipeline for a specific segmentation problem: input 
images, output polygon classifications.

* blog posts

* internal (dsc) show and tell/talk

* meta deliverables: we are trying new ways to manage this project (e.g., 
process automation, kanban). we plan to share our experiences upon completion.

### When does the project need to be completed by?

Monday, 6th October 2017.

## Data sources and data science techniques

### What data will you use for this project?

* Part of the project scope is identifying and obtaining appropriate data.
* Will need some form of pre-labelled imagery. Failing that, will need to create
a tool which will enable us to label images ourselves.

### What data science techniques is the project likely to need?

* Computer vision
* Neural networks, deep learning
* Programming: Python (modeling + pipeline), Javascript (possible visualisation)
* Geospatial tools + processing

## About you

### Who is the project lead contact for the project?

* Steven Hopkins
* Philip Stubbings
* Ioannis Tsalamanis

### Who is the day-to-day contact for the project?

* Same as above.
