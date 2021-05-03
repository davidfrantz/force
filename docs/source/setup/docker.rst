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
    docker pull \
      davidfrantz/force

This downloads the latest, fully featured FORCE (3.x) on your local machine.
You may want to do this regularly to always run the latest version of FORCE!

Check if this works:

  .. code-block:: bash

    docker run \
      davidfrantz/force force

This displays general information about FORCE, as well as the version number.

If you want to use a specific version - or the develop branch that includes the latest cutting-edge features:

  .. code-block:: bash

    # version 3.6.5
    docker run \
      davidfrantz/force:3.6.5

    # develop version
    docker run \
      davidfrantz/force:dev


.. _docker_build:

For developers: Local build
---------------------------

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder:

  .. code-block:: bash

    # This should only take a couple of minutes
    # The '-t' flag indicates how your local image will be named, in this case 'my-force'
    docker build -t my-force .


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

For the download tools, you need to share credentials between host and container.
The credentials are usually stored in ``$HOME/.boto``, ``$HOME/.scihub``, and ``$HOME/.laads``.
To make these files available, you need to attach the folder containing these files as a mounted volume, and set a Docker runtime environment variable pointing to that mounted folder location.

  .. code-block:: bash

    # --env sets the environment variable
    # this command will only print the container's FORCE_CREDENTIALS variable
    # should be:
    # FORCE_CREDENTIALS=/app/credentials
    docker run \
      -v /my/local/folder:/opt/data \
      --user "$(id -u):$(id -g)" \
      --env FORCE_CREDENTIALS=/app/credentials \
      -v $HOME:/app/credentials \
      davidfrantz/force \
      force-level1-csd -h

If you wish to enter the running container's terminal run it with an additional ``-it`` flag. 
In that case you can use this terminal just as you were on a Linux machine.

If this is too long for you, you can hide all this behind an alias (or define a function).
For an alias, add a line to ``$HOME/.bashrc`` (log off and on to take effect):

  .. code-block:: bash

    alias dforce="docker run -v /my/local/folder:/opt/data --user \"$(id -u):$(id -g)\ --env FORCE_CREDENTIALS=/app/credentials -v $HOME:/app/credentials davidfrantz/force"

After defining the alias, you can call FORCE with correct user and mounted volume - but less Docker boilerplate commands:

  .. code-block:: bash

    dforce force-level2 /opt/data/parameters.prm


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
