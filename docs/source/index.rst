.. _main:

FORCE documentation
===================

**FORCE: Framework for Operational Radiometric Correction for Environmental monitoring**

**Version 3.0**

.. image:: force.png

About
-----

FORCE is an all-in-one processing engine for medium-resolution EO image archives. FORCE uses the data cube concept to mass-generate Analysis Ready Data, and enables large area + time series applications. With FORCE, you can perform all essential tasks in a typical EO Analysis workflow, i.e. going from data to information.

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

Download the FORCE **source code** from `Github <https://github.com/davidfrantz/force>`_. It is open source and free!

Have a look at my `**Tutorials** <https://davidfrantz.github.io/tutorials>`_. Check regularly for new content.

Get help, and help others in the FORCE self-help `**Google group** <https://groups.google.com/d/forum/force_eo>`_

**Follow** the FORCE project at `ResearchGate <https://www.researchgate.net/project/FORCE-Framework-for-Operational-Radiometric-Correction-for-Environmental-monitoring>`_.

**Stay updated**, and follow me on `Twitter <https://twitter.com/d__frantz>`_

**You are using FORCE? Spread the word**, and use the `**#FORCE_EO** <https://twitter.com/search?q=%23FORCE_EO&src=recent_search_click>`_ hashtag in your tweets!



.. toctree::
   :maxdepth: 1
   :hidden:

   index.rst

   
.. toctree::
   :maxdepth: 1
   :caption: Usage Policy
   :hidden:

   policy/citation.rst
   policy/development.rst
   policy/license.rst

   
.. toctree::
   :maxdepth: 3
   :hidden:

   history/index.rst
   
   
.. toctree::
   :maxdepth: 1
   :caption: Setup
   :hidden:

   setup/requirements.rst
   setup/depend.rst
   setup/depend_opt.rst
   setup/install.rst
   
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
   :hidden:

   proj.rst
   faq.rst
   issues.rst
   refs.rst
   contact.rst


