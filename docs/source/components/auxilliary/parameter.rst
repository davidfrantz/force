.. _aux-parameter:

force-parameter
==========

force-parameter generates parameter file skeletons for each FORCE module. The skeletons also contain more in depth
descriptions for each parameter, and supported parameter values/ranges. The descriptions can be turned off to generate
more compact parameter files.

Usage
^^^^^

.. code-block:: bash

    force-parameter dir type verbose

* | ``dir`` : File path, can be either relative or absolute.
* | ``type`` : Type of parameter file that should be generated. Can be one of the following:

    +----------+-------------------------------------------+
    | LEVEL2   | :ref:`Level 2 Processing System <l2ps>`   |
    +----------+-------------------------------------------+
    | LEVEL3   | :ref:`Level 3 Processing System <level3>` |
    +----------+-------------------------------------------+
    | TSA      | :ref:`Time Series Analysis <tsa>`         |
    +----------+-------------------------------------------+
    | CSO      | :ref:`Clear-Sky Observations <cso>`       |
    +----------+-------------------------------------------+
    | L2IMP    | :ref:`Level 2 ImproPhe <l2i>`             |
    +----------+-------------------------------------------+
    | CFIMP    | :ref:`Continuous Field ImproPhe <cfi>`    |
    +----------+-------------------------------------------+
    | SMP      | :ref:`Sampling <smp>`                     |
    +----------+-------------------------------------------+
    | TRAIN    | :ref:`Train Machine Learner <aux-train>`  |
    +----------+-------------------------------------------+
    | SYNTHMIX | Synthetic Mixing                          |
    +----------+-------------------------------------------+
    | ML       | :ref:`Machine Learning <ml>`              |
    +----------+-------------------------------------------+
    | TXT      | :ref:`Texture <txt>`                      |
    +----------+-------------------------------------------+
    | LSM      | :reF:`Landscape Metrics <lsm>`            |
    +----------+-------------------------------------------+
    | LIB      | Library Completeness                      |
    +----------+-------------------------------------------+

* | ``verbose`` : If ``1``, then a long parameter file with comments will be generated.
    If ``0``, no comments will be included.

