.. _docker:

Docker support
==============

If you wish to use FORCE with docker you can do it in two ways: 

* download a **prebuilt image** from Docker hub
* create a **local build** with Dockerfile


Use prebuilt image
------------------

The easiest way to use FORCE with Docker is to use a prebuilt image pulled from Docker hub with the following command:

  .. code-block:: bash

    # This takes only a few minutes
    docker pull fegyi001/force

This downloads the latest FORCE (^3.x) with default settings (excluding SPLITS, disabled debug mode) on your local machine.
You can use FORCE like this:

  .. code-block:: bash

    docker run fegyi001/force force

Which outputs a default message showing that FORCE works indeed.

On the `Docker hub page <https://hub.docker.com/repository/docker/fegyi001/force/tags?page=1>`_ you can see multiple tags for ``fegyi001/force``. The tags refer to the version first (e.g. v3.1.0), then optionally indicates whether it is a SPLITS version and/or a debug version. E.g. if you wish to use FORCE of version 3.1.0 in debug mode & with SPLITS use this image: ``fegyi001/force:v3.1.0_splits_debug``. If you see only the version number this indicates default options (disabled SPLITS & disabled debug mode). 


Local build
-----------

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder (this will not include SPLITS!):

  .. code-block:: bash

    # This may take some time (up to 10-20 minutes).
    # The '-t' flag indicates how your local image will be called, in this case 'my-force'
    docker build -t my-force .

If you wish to enable **SPLITS** run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg splits=true .

If you wish to enable **DEBUG** mode run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg debug=true .

You can add multiple build arguments, e.g. if you wish to enable SPLITS and DEBUG mode run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg splits=true --build-arg debug=true .


After that, you can use your newly built Docker image like this:

  .. code-block:: bash

    docker run my-force force

The rest is up to you, you can do anything Docker containers support. E.g. you wish to add a volume to the container and run a ``force-level2`` command is as simple as that:

  .. code-block:: bash

    # Let's say you have a parameter file in /my/local/folder/parameters.prm
    # You map your local folder into /opt/data for your force container
    # Without it FORCE will not be able to see your local files since it is isolated
    docker run -v /my/local/folder:/opt/data my-force force-level2 /opt/data/parameters.prm

