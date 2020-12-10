.. _main:

FORCE documentation
===================

**FORCE: Framework for Operational Radiometric Correction for Environmental monitoring**

**Version 3.5.2-dev**

`Download from Github <https://github.com/davidfrantz/force>`_.

.. image:: force.png

.. note::
   The ``latest`` version of this documentation refers to the ``develop`` software branch, which includes bleeding-edge features that are not part of ``master`` yet.
   The ``stable`` version of this documentation only describes features that are present in ``master``.
   New additions to this documentation will only be included in the ``latest`` version.
   The ``stable`` version will be synced with ``latest`` whenever a new ``master`` version of FORCE is released.

.. warning::
   This documentation is not finished yet. Parts of the description are badly formatted, incomplete, incorrect or are referring to version 2.1. Please, come back later for the definitive documentation.

FORCE is ...
------------

... an all-in-one processing engine for medium-resolution Earth Observation image archives. FORCE uses the data cube concept to mass-generate Analysis Ready Data, and enables large area + time series applications. With FORCE, you can perform all essential tasks in a typical Earth Observation Analysis workflow, i.e. going from data to information.

FORCE natively supports the integrated processing and analysis of 

  * Landsat 4/5 TM, 
  * Landsat 7 ETM+, 
  * Landsat 8 OLI and 
  * Sentinel-2 A/B MSI.

Non-native data sources can also be processed, e.g. Sentinel-1 SAR data or environmental variables.


This user guide summarizes the technical aspects required to run FORCE. 

It will not give elaborated descriptions of methodology. For the methodological description, please refer to the scientific :ref:`refs`.


Related Links
-------------

**Get the source code** from `Github <https://github.com/davidfrantz/force>`_. It is open source and free!

**Learn how to use FORCE**. Have a look at the :ref:`howto`. Check regularly for new content.

**Get help**, and help others in the FORCE self-help `Google group <https://groups.google.com/d/forum/force_eo>`_

**Follow** the FORCE project at `ResearchGate <https://www.researchgate.net/project/FORCE-Framework-for-Operational-Radiometric-Correction-for-Environmental-monitoring>`_.

**Stay updated**, and follow me on `Twitter <https://twitter.com/d__frantz>`_

**You are using FORCE? Spread the word**, and use the `#FORCE_EO <https://twitter.com/search?q=%23FORCE_EO&src=recent_search_click>`_ hashtag in your tweets!




.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   about.rst


.. toctree::
   :maxdepth: 3
   :hidden:

   history/index.rst


.. toctree::
   :maxdepth: 1
   :caption: Usage Policy
   :hidden:

   policy/citation.rst
   policy/development.rst
   policy/license.rst


.. toctree::
   :maxdepth: 1
   :caption: Setup
   :hidden:

   setup/requirements.rst
   setup/depend.rst
   setup/depend_opt.rst
   setup/install.rst
   setup/docker.rst


.. toctree::
   :maxdepth: 4
   :caption: How to use the FORCE
   :hidden:

   howto/index.rst


.. toctree::
   :maxdepth: 4
   :caption: Software components
   :hidden:

   components/comp_overview.rst
   components/lower-level/index.rst
   components/higher-level/index.rst
   components/auxilliary/index.rst


.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   refs.rst
   refs-applied.rst


.. toctree::
   :maxdepth: 1
   :caption: Other
   :hidden:

   proj.rst
   issues.rst
   contact.rst

