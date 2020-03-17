.. _depend-opt:

Optional dependencies
=====================

Some FORCE functionality is dependent on additional open-source software. In order to allow a baseline installation, these dependencies are optional and are not required for the default installation. If you want to enable this functionality, you need to install following software, and follow the optional steps in section F.

* **SPLITS (Spline analysis of Time Series)**, developed by Dr. Sebastian Mader, Trier University, is a standalone software/API, which can be used to fit splines to dense time series in order to interpolate time series and to derive Land Surface Phenology.
  Some SPLITS-dependent features are implemented in force-tsa (see section VII.E). You only need to install SPLITS if you want to use these features; all other functionality of force-tsa can be used without installing SPLITS.

  SPLITS is distributed under the terms of the GNU General Public License, and can be downloaded from `<http://sebastian-mader.net/splits/>`_. SPLITS is itself dependent on GDAL (required for FORCE, see last section), Armadillo and FLTK.

  Installation instructions can be found `here <http://sebastian-mader.net/splits/>`_, or you can try the following steps (Ubuntu). Make sure to change the path to your own home directory. If the `--prefix` is omitted, SPLITS will be installed into a system-wide directory, which is preferably if the code shoould be accessed by multiple users. Note that the GDAL include directory might also be in a different path on your machine (check with `gdal-confif`).

  **Some changes in SPLITS code!!!**
  
  .. code-block:: bash

    sudo apt-get install libarmadillo-dev
    sudo apt-get install libfltk1.3-dev

    mkdir /home/MYHOME/src
    mkdir /home/MYHOME/splits

    cd /home/MYHOME/src
    wget http://sebastian-mader.net/wp-content/uploads/2017/11/splits-1.9.tar.gz
    tar -xzf splits-1.9.tar.gz
    cd splits-1.9

    ./configure --prefix=/home/MYHOME/splits CPPFLAGS="-I /usr/include/gdal" CXXFLAGS=-fpermissive
    make
    make install
    make clean

There are no known problems, except that you need to install SPLITS v. 1.9.

* **Docker**: Docker is the de facto standard to build and share containerized apps - from desktop, to the cloud. FORCE supports Docker builds (either prebuilt image on Docker hub or custom build with Dockerfile). Docker is `available for Linux, Mac and Windows <https://docs.docker.com/install/>`_. If you are unfamiliar with Docker, see `Docker website <https://www.docker.com/why-docker>`_ for more information.

