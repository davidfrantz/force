.. _hl-input:

Input type
==========

Input for HLPS must be in **datacube** format!

.. seealso:: Check out the `datacube tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains how we define a datacube.


The different :ref:`hl-submodules` either process **ARD** or **feature datasets**:


1. **ARD** are Level 2 Analysis Ready Data. 

   Alternatively, Level 3 Best Available Pixel composites can be input, too.
   They consist of a reflectance product (mostly BOA, but TOA, IMP, BAP are supported, too), and pixel-based quality information (mostly QAI, but INF is supported, too).
   These input data need to follow a strict data format, including number of bands, naming convention with time stamp, sensor etc.

   .. seealso:: Check out the `ARD tutorial <https://davidfrantz.github.io/tutorials/force-ard/l2-ard/>`_, which explains what Analysis Ready Data are, and how to use the FORCE :ref:`l2ps` to generate them..

   If multiple sensors are used, analyses are restricted to the overlapping bands:

    .. _table-sensor-bands:

    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SENSOR                         + BLUE + GREEN + RED + RE1 + RE2 + RE3 + BNIR + NIR + SWIR1 + SWIR2 + VV + VH +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND04  + Landsat 4 TM          + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND05  + Landsat 5 TM          + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND07  + Landsat 7 ETM+        + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND08  + Landsat 8 OLI         + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2A  + Sentinel-2A           + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2B  + Sentinel-2B           + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + sen2a  + Sentinel-2A           + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + sen2b  + Sentinel-2B           + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1AIA  + Sentinel-1A IW asc.   +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1BIA  + Sentinel-1B IW asc.   +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1AID  + Sentinel-1A IW desc.  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1BID  + Sentinel-1B IW desc.  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LNDLG  + Landsat legacy bands  + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2L  + Sentinel-2 land bands + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2H  + Sentinel-2 high-res   + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + R-G-B  + Visible bands         + 1    + 2     + 3   +     +     +     +      +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + VVVHP  + VV/VH Dual Polarized  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
 
2. **Feature datasets** can be anything from individual ARD datasets to external datasets like precipitation or DEM.

   Most often, features are generated by one HLPS submodule, and then used by another one, e.g. generate Spectral Temporal Metrics with :ref:`tsa`, then use these outputs as features in :ref:`ml`.
   The most important constraint is: HLPS only knows 16bit signed input, thus if you import external data, you need to scale accordingly.


