.. _tut-virtual-datacube:

Building a Virtual Datacube
===========================

**How to merge cubes**

This tutorial explains how multiple datacubes can be integrated into one single, "virtual" datacube.

.. admonition:: Info

   *This tutorial uses FORCE v. 3.8.02*


Why?
----

FORCE makes heavy use of the data cube concept.
This includes two main points:

1. All data are in the **same coordinate system**, which should be valid for a large regional extent (e.g. a continental projection).
2. The data are organized in regular, non-overlapping **tiles**.

If the same datacube parameters are used, interoperability between datasets is ensured!

However, FORCE also requires that all data that are to be analyzed together, need to be in the same datacube.

There may be reasons when this requirement cannot be met, or when it is not desired. 

As an example, a FORCE datacube is hosted on `CODE-DE <https://code-de.org/>`_. 
This datacube contains Level 2 ARD of all Landsat and Sentinel-2 data that cover Germany, and is continuously updated.
It can be used by any person or institution, which has access to the platform (most public players in Germany have free access).
For obvious reasons, however, users do not have write permissions within the datacube.
Therefore, if users want to combine this datacube with their own data (let's assume a PlanetScope datacube), they either need to copy data from the cube to their working environment - or need to become creative.

Another reason could be that different data source should be held separately, or that storage limitations prevent to have a big cube in the same physical drive.


The solution: a virtual cube
----------------------------

The solution is pretty simple: in GDAL, there is the super-convenient VRT format (GDAL Virtual Format). 
The VRT format is essentially an XML-based format that maps its attributes and geometries to that of an underlying data source of any GDAL-supported raster format. 
In short, it is a lightweight translator file, which just contains links to the original data.

EXAMPLE here.


This format is already used for quite a while throughout FORCE, e.g., when creating mosaics from several tiles.

It is also long possible to use ``force-higher-level`` on VRT datasets - although this is probably a lesser known feature.
That said, a simple and straightforward solution to creating a joint datacube from two or multiple datacubes is to generate a **virtual datacube**.
This datacube has the same structure as the original datacubes, but just contains data in the form of VRT files.
Of course, the links in the VRTs should be functional, and the individual datacubes need to follow the same datacube definition (projection, grid size and origin).

Since FORCE 3.8.02, there is a new tool, ``force-virtual-datacube``, which helps you build such a database.

The usage is simple. 
Let's assume we have two datacubes on different mount points, e.g.:

/mnt/s2/ard and /mnt/planet/ard

Let's also assume that we don't have write permissions in none of them.
And for demonstration purposes, let's also assume that the file format differs.

But we can write here:

/data/my-project

force-virtual-datacube -p '*.tif' /mnt/s2/ard /data/my-project/ard
force-virtual-datacube -p '*.jp2' /mnt/planet/ard /data/my-project/ard

