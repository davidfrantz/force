.. _docker:

Docker / Singularity support
============================

If you wish to use FORCE with docker you can do it in two ways: 

* download a **prebuilt image** from Docker hub
* create a **local build** with Dockerfile

If you wish to use FORCE with Singularity, please see the instructions :ref:`below <singularity>`. 


Ready-to-go: Pull a pre-built image
-----------------------------------

The easiest way to use FORCE with Docker is to use a prebuilt image pulled from Docker hub with the following command:

  .. code-block:: bash

    # This takes only a few minutes
    docker pull davidfrantz/force

This downloads the latest, fully featured FORCE (3.x) on your local machine.
You can use FORCE like this:

  .. code-block:: bash

    docker run davidfrantz/force force

Which displays general information about FORCE, as well as the version number.


For developers: Local build
---------------------------

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder:

  .. code-block:: bash

    # This may take some time (up to 10-20 minutes).
    # The '-t' flag indicates how your local image will be called, in this case 'my-force'
    docker build -t my-force .

There are optional build parameters for enabling/disabling SPLITS and/or DEBUG mode. By default SPLITS is enabled and DEBUG mode is disabled.

If you wish to disable **SPLITS** run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg splits=false .

If you wish to enable **DEBUG** mode run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg debug=true .

You can add multiple build arguments, e.g. if you wish to disable SPLITS and enable DEBUG mode run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg splits=false --build-arg debug=true .


Usage
-----

After downloading or building your own image, you can run it as a container like this:

  .. code-block:: bash

    # using the prebuilt image
    docker run davidfrantz/force force

    # using a custom built image
    docker run my-force force

The rest is up to you, you can do anything Docker containers support. E.g. you wish to add a volume to the container and run a ``force-level2`` command is as simple as that:

  .. code-block:: bash

    # Let's say you have a parameter file in /my/local/folder/parameters.prm
    # You map your local folder into /opt/data for your force container
    # Without it FORCE will not be able to see your local files since it is isolated
    docker run -v /my/local/folder:/opt/data davidfrantz/force force-level2 /opt/data/parameters.prm

If you wish to enter the running container's terminal run it with the ``-it`` flag. In that case you can use this terminal just as you were on a Linux machine.

  .. code-block:: bash

    docker run -it -v /my/local/folder:/opt/data davidfrantz/force
  

User credentials
----------------

If you have ``.scihub`` and ``.laads files`` on your local machine and you wish them to be used by FORCE in Docker you should attach the folder containing these files as a mounted volume, and set a Docker runtime environment variable pointing to that mounted folder location. 
This sounds complicated, but it really isn't:

  .. code-block:: bash

    # --env sets the environment variable
    # this command will only print the container's FORCE_CREDENTIALS variable
    # is should result this:
    # FORCE_CREDENTIALS=/app/credentials
    docker run --env FORCE_CREDENTIALS=/app/credentials -v /path/to/credentials/folder/on/your/machine:/app/credentials davidfrantz/force env | grep FORCE_CREDENTIALS


.. _singularity:

Singularity
-----------

The FORCE Docker images can be simply run using Singularity.

The simplest way is to directly run the Docker image:

 .. code-block:: bash

    singularity exec docker://davidfrantz/force:latest force

This will automatically pull the Docker image from Docker Hub, and convert it to a Singularity image.
The image can be updated by regularly doing:

.. code-block:: bash

    singularity pull -F docker://davidfrantz/force:latest

You can also create a local copy of the image by explicitly doing the conversion:

.. code-block:: bash

    singularity build force.sif docker://davidfrantz/force:latest

    singularity exec force.sif force
