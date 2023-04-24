.. _depend:

Dependencies
============

There are a number of required open-source dependencies. The suggested installation procedures are tested with ``Ubuntu 18.04 LTS`` and ``Ubuntu 20.04 LTS``. 

FORCE can also be installed on other Liunx distributions (e.g. CentOS). The installation of packages generally works similarly, but some adaptation might be needed.

.. note::
   As FORCE is being developed further, dependencies are growing, and installation becomes more complex.
   Thus, we suggest to consider using FORCE with Docker or Singularity, see :ref:`docker`.
   This allows you to skip the complete installation, and to always use the latest FORCE version.


* **GNU parallel** is used for some parallelization tasks.
  We developed the code using version 20140322.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install parallel


  Parallel has to use ``--gnu mode``, not ``--tollef``. If ``--tollef`` is your default (occurred on older installations), fix this permanently by deleting the ``--tollef`` flag in ``/etc/parallel/config``. Refer to the mainpage of parallel for details.
  Parallel will display a citation request. To silence this citation notice run this code here once:
  
  .. code-block:: bash

    parallel --bibtex

* The **GDAL API and commandline tools** are used for I/O and various image processing tasks.
  We developed the code using version 2.2.2.
  Note that different Ubuntu versions come with different GDAL versions.

  * Ubuntu 18.04 LTS:
    The software can be installed with:

    .. code-block:: bash

      sudo apt-get install libgdal-dev gdal-bin python-gdal

  * Ubuntu 16.04 LTS (and before): 
    Upgrade GDAL by adding the unstable UbuntuGIS repository:
  
    .. code-block:: bash

      sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
      sudo apt-get update

    If GDAL was already installed, upgrade:
  
    .. code-block:: bash

      sudo apt-get dist-upgrade

    If not, install with 

    .. code-block:: bash
  
      sudo apt-get install libgdal1-dev gdal-bin python-gdal

  *There are known problems with earlier releases (< 1.10.0). FORCE < 3.5 should not be used with GDAL >= 3.0.
  However, the reporting of errors and warnings differs between versions, and GDAL may report many non-critical errors to stderr (e.g. ``ERROR 6 - not supported``, please refer to the GDAL error code description whether these are critical errors or just warnings that can be ignored). Please note that GDAL is a very dynamic development, therefore it is hard to develop applications that cover all cases and exceptions in all possible GDAL versions and builds. If you come across a GDAL version that does not work, please inform us.*

* The **GSL library** is used for optimization purposes.
  We developed the code using version 1.15.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install libgsl0-dev

* The **cURL library** is used to download MODIS water vapor data.
  We developed the code using version 7.22.0.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install curl

* **unzip** is used to extract Sentinel-2 data.
  We developed the code using version 6.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install unzip

* **lockfile-progs** is used to place a temporary lock on file queues.
  The utility is already included in some distributions.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install lockfile-progs

  *There is a known problem with CIFS mounted network drives. You may get a lot of warnings like ``lockfile creation failed: exceeded maximum number of lock attempts``. You can ignore these warnings; they are no fatal errors. But you might want to inspect the file queue after Level 2 processing, as there is a minor possibility that there were some conflicts due to parallel write attempts: a few images might not have been switched from ``QUEUED`` to ``DONE`` status.*

* **rename** is used to rename files.
  The tool is missing in new Ubuntu distributions (Ubuntu > 17.10). The software can be installed with:

  .. code-block:: bash

    sudo apt-get install rename

* **python3** is used by a couple of auxilliary scripts.
  python3 should already be installed. If not, you can install like this:

  .. code-block:: bash

    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get install python3.8 python3-pip python3-dev
    echo 'alias python=python3' >> ~/.bashrc
    echo 'alias pip=pip3' >> ~/.bashrc

* Some **python packages** are needed:

  .. code-block:: bash

    pip install numpy gsutil git+https://github.com/ernstste/landsatlinks.git

* **pandoc** is used to convert from markdown to html.
  The software can be installed with:

  .. code-block:: bash

    sudo apt-get install pandoc

* **R** is used by a couple of auxilliary scripts.

  .. code-block:: bash

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -sc)-cran40/"
    sudo apt-get install r-base

* Some **R packages** are needed:

  .. code-block:: bash

    Rscript -e 'install.packages("rmarkdown", repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("plotly",    repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("stringi",   repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("knitr",     repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("dplyr",     repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("snow",      repos="https://cloud.r-project.org")'
    Rscript -e 'install.packages("snowfall",  repos="https://cloud.r-project.org")'

* **OpenCV** is used for machine learning and image processing tasks.
  We developed the code using OpenCV v. 4.1. 
  The installation process might need some more dependencies, e.g. ``cmake``.
  The software needs to be installed manually. 
  See the `installation instructions <https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html>`_ or try following recipe:

  .. code-block:: bash

     mkdir -p ~/src/opencv
     cd ~/src/opencv
     wget https://github.com/opencv/opencv/archive/4.1.0.zip
     unzip 4.1.0.zip
     cd opencv-4.1.0
     mkdir build
     cd build
     cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
     make -j
     sudo make install
     make clean

* **aria2** is used to download Landsat Level 1 product bundles with ``force-level1-landsat``.

  .. code-block:: bash

    sudo apt install aria2