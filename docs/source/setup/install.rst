.. _install:

Installation
============

Installation instructions
-------------------------

FORCE is mostly written in C/C++: Thus, it needs to be compiled. Administrator rights are not necessarily required, unless you want to install to the system-wide search path (e.g. to make it available to multiple users).

This recipe assumes that you have installed all dependencies. 
Please refer to the ``Dockerfile`` within the FORCE repository for figuring out the dependencies.

We are however strongly suggesting to consider using FORCE with Docker or other container technologies, see :ref:`docker`.


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

    force-info

  If the program cannot be found, you will likely need to add this directory to your search path ``$PATH`` (see `here <https://opensource.com/article/17/6/set-path-linux>`_). This might happen if you have used a custom installation directory.
  
  
Installation of the development version
---------------------------------------

To install FORCE with all "bleeding-edge features", consider to use the develop version.

* After point (3), change to the develop branch, then proceed with (4).
  Note that `BINDIR` defaults to ``/develop`` on this branch. Not installing to ``/usr/local/bin`` (which is the default for the master branch) might make sense in the case you want to have both the master and develop versions installed.
  You might want to change `BINDIR` to a directory that suits you (e.g. ``/usr/local/bin`` or a local directory).

  .. code-block:: bash

    git checkout -b develop
    git pull origin develop

  If you are running different versions - or want to make sure to always use the latest version - consider using pre-built Docker images, see :ref:`docker`.


Installation in DEBUG mode
--------------------------

Follow these steps before step 3 in the installation instruction:

a) Enable DEBUG in FORCE

    .. code-block:: bash
    
      cd ~/src/force
      ./debug.sh enable

b) Proceed with the installation of FORCE

  