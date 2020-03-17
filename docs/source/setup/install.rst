.. _install:

Installation
============

FORCE is mostly written in C/C++: This, it needs to be compiled. Administrator rights are not necessarily required, unless you want to install to the system-wide search path (e.g. to make it available to muptiple users).

1. Go to the directory, where you usually store source code, e.g.

  .. code-block:: bash

    cd ~/src

2. Pull from Github

  .. code-block:: bash

    git clone https://github.com/davidfrantz/force.git

3. Go to the force directory

  .. code-block:: bash

    cd force

4. Edit the Makefile with the text editor of your choice, e.g.

  .. code-block:: bash

    vi Makefile

  * `BINDIR` is the directory used for installing the software. 

    * By default, this is ``/usr/local/bin``. This is a system-wide search path. If you install to this directory, all users on this machine can use the code. You need admin rights to install to this directory. 

    * Alternatively, you can specify a private directory, e.g. ``/home/YOURNAME/bin``. In this case, the code is only available to you, but you do not need admin rights for the installation.

  * The next block specifies the location of libraries and headers. It might be possible that you need to make some changes here.

  * The rest of the Makefile should be OK. Only edit anything if you know what you are doing.

5. Compile the code

  .. code-block:: bash

    make -j

6. Install the code. If you are installing to a system directory, include ``sudo`` to install with admin rights. If you are installing to a private directory, remove the ``sudo``.

  .. code-block:: bash

    sudo make install

7. Test if it was installed correctly:

  .. code-block:: bash

    force

  If the program cannot be found, you will likely need to add this directory to your search path ``$PATH`` (see `here <https://opensource.com/article/17/6/set-path-linux>`_). This might happen if you have used a custom installation directory.


Installation with optional software
-----------------------------------

* Install with SPLITS.

  Follow these steps before step 3 in the installation instruction:

  a) Install SPLITS, see :ref:`depend-opt`

  b) In ``src/cross-level/const-cl.h``, uncomment the SPLITS preprocessor definition ``#define SPLITS``.

     .. code-block:: bash
     
       cd ~/src/force
       sed -i -e 's%^[/]*\(#define SPLITS\)%\1%' src/cross-level/const-cl.h
  
  c) Edit the Makefile. 
     ``SPLITS`` names the directories, where SPLITS header files and library are installed.     
     This line needs to be uncommented, as well as the ``LDSPLITS`` line.
  
     .. code-block:: bash

       sed -i -e 's%^[#]*\(SPLITS\)%\1%' Makefile
       sed -i -e 's%^[#]*\(LDSPLITS\)%\1%' Makefile

     If SPLITS was installed to a different location, you need to adjust this to your needs.
     
     .. code-block:: bash

       vi Makefile
     
     Example: 
     
     ``SPLITS=-I/usr/local/include/splits -L/usr/local/lib -Wl,-rpath=/usr/local/lib``

  d) Proceed with the installation of FORCE


Installation with Docker
------------------------

* Use prebuilt image

  The easiest way to use FORCE with Docker is to use a prebuilt image pulled from `Docker hub <https://hub.docker.com/>`_ with the following command:
  
  ``docker pull fegyi001/force:latest``

  This downloads a fully featured FORCE v3.0 on your local machine including SPLITS.
  You can use FORCE like this:

  ``docker run fegyi001/force force``

* Local build

  If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder:

  ``docker build -t my-force .``

  After that, you can use your newly built Docker image like this:

  ``docker run my-force force``

For more details visit the Readme in the `docker` subfolder.