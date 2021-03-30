.. _docker:

Docker / Singularity support
============================

If you wish to use FORCE with docker you can do it in two ways: 

* download a **prebuilt image** from Docker hub
* create a **local build** with Dockerfile

If you wish to use FORCE with Singularity, please see the instructions :ref:`below <singularity>`. 


.. _docker_pull:

Ready-to-go: Pull a pre-built image
-----------------------------------

The easiest way to use FORCE with Docker is to use a prebuilt image pulled from Docker hub with the following command:

  .. code-block:: bash

    # This takes only a few minutes
    docker pull davidfrantz/force

This downloads the latest, fully featured FORCE (3.x) on your local machine.
You may want to do this regularly to always run the latest version of FORCE!

Check if this works:

  .. code-block:: bash

    docker run davidfrantz/force force

Which displays general information about FORCE, as well as the version number.


.. _docker_build:

For developers: Local build
---------------------------

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder:

  .. code-block:: bash

    # This may take some time (up to 10-20 minutes).
    # The '-t' flag indicates how your local image will be called, in this case 'my-force'
    docker build -t my-force .

There are optional build parameters for enabling/disabling SPLITS and/or DEBUG mode. By default SPLITS is enabled and DEBUG mode is disabled.


.. _docker_use:

Usage
-----

After downloading or building your own image, you can run it as a container like this:

  .. code-block:: bash

    # using the prebuilt image
    docker run davidfrantz/force force

    # using a custom built image
    docker run my-force force

The Docker container is isolated from your host, thus FORCE will not be able to see your local files.
To share a volume, e.g. for input/output data, you can map a local folder to a folder within the container:

  .. code-block:: bash

    docker run \
      -v /my/local/folder:/opt/data \
      davidfrantz/force \
      force-level2 /opt/data/parameters.prm

The user within the container is different than on your host.
To avoid issues with file permissions, you can map your local user to the user within the container:

  .. code-block:: bash

    docker run \
      -v /my/local/folder:/opt/data \
      --user "$(id -u):$(id -g)" \
      davidfrantz/force \
      force-level2 /opt/data/parameters.prm

If this is too long for you, you can define an alias in ``~/.bashrc``:

  .. code-block:: bash

    alias dforce="docker run -v /my/local/folder:/opt/data --user \"$(id -u):$(id -g)\" davidfrantz/force"

Then, you can call FORCE with correct user and mounted volume like this:

    dforce force-level2 /opt/data/parameters.prm


If you wish to enter the running container's terminal run it with the ``-it`` flag. 
In that case you can use this terminal just as you were on a Linux machine.

  .. code-block:: bash

    docker run \
      -v /my/local/folder:/opt/data \
      --user "$(id -u):$(id -g)" \
      davidfrantz/force

If you want to use a specific version - or the develop branch that includes the latest cutting-edge features:

  .. code-block:: bash

    docker run \
      davidfrantz/force:3.6.5

    docker run \
      davidfrantz/force:dev


.. _docker_credentials:

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
