.. _tut-lcf:

Land Cover Fraction Mapping
=========================

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

**How to use FORCE submodules in a workflow to map sub-pixel fractions of land cover with synthetically mixed training data**

This tutorial demonstrates a workflow that uses a series of submodules of the FORCE Higher Level Processing system (HLPS) to map sub-pixel fractions of land cover with Sentinel-2 imagery.

While spectral unmixing has been used for decades in different contexts, with different sensor types and different methods, this tutorial walks through a regression-based approach using synthetically mixed training data as presented in `Okujeni et al. (2013) <https://doi.org/10.1016/j.rse.2013.06.007>`_.

.. admonition:: Info

   *This tutorial uses FORCE v. 3.7.9*

The Workflow
-----------------------------------

The workflow in this tutorial uses a series of submodules of the FORCE Higher Level Processing system (HLPS) to map sub-pixel fractions of land cover with Sentinel-2 imagery and synthetically mixed training data. 

Some of these submodules have been described in other places of the FORCE documentation, and entire tutorials have been dedicated to others (respective links will be given where applicable).

This tutorial illustrates the potential of FORCE to be used along the complete image processing chain, from downloading and pre-processing image acquisitions to producing meaningful spatial data. Its chapters correspond to the seven steps (and two optional steps) of the following workflow. 

This workflow is reproducible, as all commands, parameter files and intermediate data will be provided. Parameter files will be directly available for download throughout the workflow, while intermediate data will only be available in a data repository due to file size (Download data from Zenodo, doi: xxxxx)

.. figure:: img/tutorial-lcf-workflow.jpg
   :height: 400

   *FORCE HLPS workflow for land cover fraction mapping with regression-based unmixing and syhnthetically mixed training data* |copy| *Franz Schug*