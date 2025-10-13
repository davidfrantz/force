.. _requirements:

Requirements
============

Hardware
--------

FORCE is intended for mass processing of medium-resolution satellite image archives.
Thus, the hardware requirements are closely tied to the data volume of the envisaged project (space / time), as well as to the type of sensor (RAM / storage requirements for Sentinel-2 are higher than Landsat).
Although the framework can also be used to process single (or a few) images, we generally recommend to use multi-CPU server systems. FORCE has been run on a multitude of architectures, from individual PCs to servers, cloud-based VMs, up to HPC supercomputers.

FORCE is command line software; a GUI is not provided.

It is advised to monitor RAM usage, and we recommend to decrease the number of CPUs if swapping occurs.
The software installation itself is small, but the disk should be large enough for the envisaged project (commonly a couple of Terabytes); note that the software does not check for free space.
Internet access is needed for some optional functionality.


Operating system
----------------

The software was developed and tested under Ubuntu LTS operating systems.
Note that we do not provide support for migrating the code to any other OS.

The code can also be installed in other Linux distributions, e.g. CentOS.
However, installation instructions are only given for Ubuntu.

FORCE can (and we urgently recommed) be run with Docker (see :ref:`docker`) or similar container technologies.

We were able to successfully install and run FORCE on the Windows 10 subsystem for Linux, too.
This can be enabled in Windows developer mode (`see this blogpost <https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/>`_).
Please note that this procedure is in beta stage and might not be as stable as running FORCE on a real Linux system.

