.. _aux-train:


force-train
===========

:ref:`Machine Learning <ml>` needs a modelset for each response variable, which can be generated as \*.xml file with force-train.
A :ref:`parameter file <train-param>` is mandatory for the Machine Learning Trainer.

Usage
^^^^^

.. code-block:: bash
   
   force-train [-h] [-v] [-i] parameter-file

   -h  = show this help
   -v  = show version
   -i  = show program's purpose

   Positional arguments:
   - 'parameter-file': TRAIN parameter file

.. toctree::
   :hidden:
   :maxdepth: 2

   train-param.rst
   
