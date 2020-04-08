.. _l2ps:

Level 2 Processing System
=========================

The FORCE Level 2 Processing System (FORCE L2PS) generates harmonized, standardized, geometrically and radiometrically consistent Level 2 products with per-pixel quality information, i.e. Analysis Ready Data (ARD).

For a description of the method, please refer to the :ref:`refs`, especially

* http://doi.org/10.3390/rs11091124
* http://doi.org/10.1109/TGRS.2016.2530856
* http://doi.org/10.1016/j.rse.2018.04.046
* http://doi.org/10.3390/rs10020352
* http://doi.org/10.3390/rs11030257


L2PS pulls each enqueued Level 1 image and processes it to ARD specification. 
This includes cloud and cloud shadow detection, potentially co-registration, radiometric correction and data cubing.

Each image (box in Figure 1) is processed independently using multiprocessing and optionally multithreading. 
The pipeline is memory resident to minimize input/output (I/O), i.e. input data are read once, and only the final, gridded data products are written to disc.
The data generated with this module are the main input for the :ref:`hlps` component.


.. image:: L2PS.jpg

**Figure 1.** FORCE Level 2 Processing System (L2PS) workflow.  


FORCE L2PS consists of two main executables, and a wrapper script that acts as a bridge between them. 
For the majority of users, it is recommended to use :ref:`level2-bulk`. 
However, for specific purposes (e.g. testing/debugging, or if you want/need to implement your own job scheduler), the expert user may want to use :ref:`force-core` or :ref:`force-wrapper` directly.


+--------------------+-----------------------+--------------------+
+ :ref:`level2-bulk` + :ref:`level2-wrapper` + :ref:`level2-core` +
+--------------------+-----------------------+--------------------+


**A glimpse of what you get:**

.. image:: ARD-L2.jpg

**Figure 2.** Data Cube of Landsat 7/8 and Sentinel-2 A/B Level 2 ARD. A two-month period of atmospherically corrected imagery acquired over South-East Berlin, Germany, is shown here.

.. toctree::
   :hidden:
   :maxdepth: 2

   level2.rst
   l2ps_.rst
   l2ps.rst
   param.rst
   format.rst
   depend.rst

