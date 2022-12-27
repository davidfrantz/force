.. _tut-lcf:

Land Cover Fraction Mapping
=========================

.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN

**How to use FORCE submodules in a workflow to map sub-pixel fractions of land cover with synthetically mixed training data**

This tutorial demonstrates a workflow that uses a series of submodules of the FORCE Higher Level Processing system (HLPS) to map sub-pixel fractions of land cover with Sentinel-2 imagery.

While spectral unmixing has been used for decades in different contexts, with different sensor types and different methods, this tutorial walks through a regression-based approach using synthetically mixed training data as presented in `Okujeni et al. (2013) <https://doi.org/10.1016/j.rse.2013.06.007>`_.

.. admonition:: Info

   *This tutorial uses FORCE v. 3.7.9*
