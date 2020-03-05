.. _issues:

Known Issues
============

There are a couple of known (and surely unknown) issues in FORCE, which can affect image quality.

Please note that FORCE was developed for scientific purposes to fulfill the research needs of the author team.
FORCE started as a project to correct Landsat imagery in Southern Africa for land cover change research.
In the meantime, it was further developed, new modules were added, support for Sentinel-2 was added and functionality for other environments were incorporated (e.g. AOD estimation over dense dark vegetation or improved topographic correction).

FORCE was tested and validated in a variety of environments and settings, and was distributed in the hope that it will be helpful to fulfill your research needs, too.
However, there are still a number of issues, which we didn’t have time to solve yet.
Note that some of the ‘problems’ mentioned below are just messages of caution; in any case, we advise you to make use of the quality flags.

You are invited to develop solutions for the problems mentioned below, and to report other unknown issues, mistakes and bugs.
I would welcome if you would pass back improvements, in which case I will try my best to find time to review and incorporate these changes into the main build (:ref:`contact` me).

 
* **AOD estimation in bright landscapes**

  FORCE L2PS estimates AOD over dark targets.
  Thus, if there are none or few, AOD estimation becomes less reliable or might fail completely.
  If there is not a single valid dark target, a global fallback value is used, which might not represent the actual conditions very well.
  Use these scenes with caution, and take a look at the AOD quality flags.

  You can use externally provided fallback values to counter this.

  Note that water bodies might not be suitable per se; if they are very turbid, they can even be brighter than the land surface in the visible bands; AOD cannot be estimated from these targets.

  Note that the presence of vegetation is also not sufficient per se; the vegetation needs to be dark and dense.
 
* **AOD estimation over the ocean**

  AOD estimation over the ocean works well in general.
  However, white-caps are problematic as they increase water reflectance, which yields higher AOD values.
  A test is made if the water body is unrealistically bright, in these cases the water body is rejected; this may happen over large lakes or ocean.

* **Environment correction**

  The environment corrects for adjacency effects, i.e. it removes the part of the radiation that comes from neighboring surface elements.
  In some cases, the correction may be too strong.
  The result can be sub-zero pixels, e.g. in the visible wavelengths.
  It was mainly observed over darker-than-usual pixels, e.g. topographic cast shadows, cloud shadows, or forest shadows (because the environment reflectance is estimated over unshaded pixels).
  Because of spatial resolution, it is more pronounced in Sentinel-2 imagery.
  Environment correction will also aggravate problems caused by a potential overestimation of AOD
  Be sure to check the sub-zero flag when using this option.
 
* **Topographic correction of poorly illuminated areas**

  The implemented topographic correction is not intended to correct hard shadows.
  Areas with illumination angle > 80° are not expected to be corrected reliably.
  Areas with illumination angle > 90° (self-shadow) are not corrected.
  Make sure to look at the illumination flags.

 
* **‘Flat’ vegetation spectra**

  Flat vegetation spectra (in the visible bands) were observed under low sun elevations.
  This may be related to radiative transfer (is based on 5S), AOD overestimation or to some bug.
  We could not yet confirm the reason for this.

