.. _docker:

Docker support
==============

If you wish to use FORCE with docker you can do it in two ways: 

* download a prebuilt image from Docker hub
* create a local build with Dockerfile


Use prebuilt image
------------------

The easiest way to use FORCE with Docker is to use a prebuilt image pulled from Docker hub with the following command:

  .. code-block:: bash

    # This takes only a few minutes
    docker pull fegyi001/force:latest

This downloads the latest, fully featured FORCE (^3.x) on your local machine including SPLITS.
You can use FORCE like this:

  .. code-block:: bash

    docker run fegyi001/force force

Which outputs a default message showing that FORCE works indeed.


Local build
-----------

If you wish to build a Docker image instead of using the prebuilt version you can do it with the following steps from the root folder (this will not include SPLITS!):

  .. code-block:: bash

    # This may take some time (up to 10-20 minutes).
    # The '-t' flag indicates how your local image will be called, in this case 'my-force'
    docker build -t my-force .

If you wish to enable SPLITS run the build with the following command:

  .. code-block:: bash

    docker build -t my-force --build-arg splits=true .

After that, you can use your newly built Docker image like this:

  .. code-block:: bash

    docker run my-force force

The rest is up to you, you can do anything Docker containers support. E.g. you wish to add a volume to the container and run a ``force-level2`` command is as simple as that:

  .. code-block:: bash

    # Let's say you have a parameter file in /my/local/folder/parameters.prm
    # You map your local folder into /opt/data for your force container
    # Without it FORCE will not be able to see your local files since it is isolated
    docker run -v /my/local/folder:/opt/data my-force force-level2 /opt/data/parameters.prm