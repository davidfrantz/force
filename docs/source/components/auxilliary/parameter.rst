.. _aux-parameter:

force-parameter
===============

force-parameter generates parameter file skeletons for each FORCE module. The skeletons also contain more in depth
descriptions for each parameter, and supported parameter values/ranges. The descriptions can be turned off to generate
more compact parameter files.

Usage
^^^^^

.. code-block:: bash

    force-parameter [-h] [-v] [-i] [-m] [-c] parameter-file module

    optional:
    -h  = show this help
    -v  = show version
    -i  = show program's purpose
    -m  = show available modules
    -c  = generate more compact parameter files without comments

    Positional arguments:
    - 'parameter-file': output file
    - 'module': FORCE module. Use -m to show available modules.



Currently available modules are:

+----------+-------------------------------------------+
| LEVEL2   | :ref:`Level 2 Processing System <l2ps>`   |
+----------+-------------------------------------------+
| LEVEL3   | :ref:`Level 3 Processing System <level3>` |
+----------+-------------------------------------------+
| TSA      | :ref:`Time Series Analysis <tsa>`         |
+----------+-------------------------------------------+
| CSO      | :ref:`Clear-Sky Observations <cso>`       |
+----------+-------------------------------------------+
| UDF      | Plug-In User Defined Functions            |
+----------+-------------------------------------------+
| L2IMP    | :ref:`Level 2 ImproPhe <l2i>`             |
+----------+-------------------------------------------+
| CFIMP    | :ref:`Continuous Field ImproPhe <cfi>`    |
+----------+-------------------------------------------+
| SMP      | :ref:`Sampling <smp>`                     |
+----------+-------------------------------------------+
| TRAIN    | :ref:`Train Machine Learner <aux-train>`  |
+----------+-------------------------------------------+
| SYNTHMIX | :ref:`Synthetic Mixing <aux-synthmix>`    |
+----------+-------------------------------------------+
| ML       | :ref:`Machine Learning <ml>`              |
+----------+-------------------------------------------+
| TXT      | :ref:`Texture <txt>`                      |
+----------+-------------------------------------------+
| LSM      | :reF:`Landscape Metrics <lsm>`            |
+----------+-------------------------------------------+
| LIB      | Library Completeness                      |
+----------+-------------------------------------------+
