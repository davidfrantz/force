.. _main:

FORCE documentation
===================

**FORCE: Framework for Operational Radiometric Correction for Environmental monitoring**

**Version 3.0**

.. image:: force.png

About
-----

FORCE is an all-in-one processing engine for medium-resolution EO image archives. FORCE uses the data cube concept to mass-generate Analysis Ready Data, and enables large area + time series applications. With FORCE, you can perform all essential tasks in a typical EO Analysis workflow, i.e. to go from data to information.

FORCE natively supports the integrated processing and analysis of 

  * Landsat 4/5 TM, 
  * Landsat 7 ETM+, 
  * Landsat 8 OLI and 
  * Sentinel-2 A/B MSI.

Non-native data sources can also be processed, e.g. Sentinel-1 SAR data or environmental variables.


This user guide summarizes the technical aspects required to run FORCE. 

It will not give elaborated descriptions of methodology. For the methodological description, please refer to the scientific publications (:ref:`refs`).


.. toctree::
   :maxdepth: 1
   :hidden:

   links.rst

   
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


