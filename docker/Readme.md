# FORCE v3.0 with Docker

## Use prebuilt image

The easiest way to use FORCE with Docker is to use a prebuilt image pulled from [Docker hub](https://hub.docker.com/) with the following command:

```sh
# This takes only a few minutes
docker pull fegyi001/force:latest
```

This downloads the latest, fully featured FORCE (^3.x) on your local machine including SPLITS.
You can use FORCE like this:

```sh
docker run fegyi001/force force
```

Which outputs a default message showing that FORCE works indeed (the exact version may differ):

```sh
##########################################################################

Hello user! You are currently running FORCE v. 3.0.1
Framework for Operational Radiometric Correction for Environmental monitoring
Copyright (C) 2013-2020 David Frantz, david.frantz@geo.hu-berlin.de
With active code contributions from
  Franz Schug, franz.schug@geo.hu-berlin.de

FORCE is free software under the terms of the GNU General Public License as published by the Free Software Foundation, see <http://www.gnu.org/licenses/>.

Thank you for using FORCE! This software is being developed in the hope that it will be helpful for you and your work.

However, it is requested that you to use the software in accordance with academic standards and fair usage. Without this, software like FORCE will not survive. This includes citation of the software and the scientific publications, proper acknowledgement in any public presentation, or an offer of co-authorship of scientific articles in case substantial help in setting up, modifying or running the software is provided by the author(s).

At minimum, the citation of following paper is requested:
Frantz, D. (2019). FORCE—Landsat + Sentinel-2 Analysis Ready Data and Beyond. Remote Sensing, 11, 1124

Each FORCE module will generate a "CITEME" file with suggestions for references to be cited. This list is based on the specific parameterization you are using.

The documentation is available at force-eo.readthedocs.io

Tutorials are available at davidfrantz.github.io/tutorials

FORCE consists of several components:
+ force-level1-landsat   Maintenance of Level 1 Landsat data pool
+ force-level1-sentinel2 Download + maintenance of Level 1 Sentinel-2 data pool
+ force-parameter        Generation of parameter files
+ force-level2           Level 2 processing of image archive
+ force-l2ps             Level 2 processing of single image
+ force-higher-level     Higher level processing (compositing, time series analysis, ...)
+ force-train            Training (and validation) of Machine Learning models
+ force-qai-inflate      Inflate QAI bit layers
+ force-tile-finder      Find the tile, pixel, and chunk of a given coordinate
+ force-tabulate-grid    Extract the processing grid as shapefile
+ force-cube             Ingestion of auxiliary data into datacube format
+ force-pyramid          Generation of image pyramids
+ force-mosaic           Mosaicking of image chips

##########################################################################
```

## Local build

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps **from the root folder**:

```sh
# This may take some time (up to 10-20 minutes).
# The '-t' flag indicates how your local image will be called, in this case 'my-force'
docker build -t my-force .
```

After that, you can use your newly built Docker image like this:

```sh
docker run my-force force
```

The rest is up to you, you can do anything Docker containers support. E.g. you wish to add a volume to the container and run a `force-level2` command is as simple as that:

```sh
# Let's say you have a parameter file in /my/local/folder/parameters.prm
# You map your local folder into /opt/data for your force container
# Without it FORCE will not be able to see your local files since it is isolated
docker run -v /my/local/folder:/opt/data my-force force-level2 /opt/data/parameters.prm
```
  