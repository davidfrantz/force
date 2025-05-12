.. _vdev:

Develop version
===============

- **FORCE L2PS**

  - When merging chips of the same day/sensor to avoid data redundancy, the merging 
    strategy for the QAI files has been revised. Before, it was just updating the images,
    i.e., merely overlaying them on top of each other. Now, the QAI files are merged feature-based
    on a custom logic such that the most restrictive QAI value is kept. This ensures that reproducibility
    is guaranteed, independent of the order in which the chips are merged. 

- **FORCE HLPS**

  - in ``force-higher-level``, UDF sub-module:
    a new feature was added to the UDF module, which allows users to add auxiliary products
    to the data array that is passed to the UDF. 
    The user can specify which auxiliary products to use in the configuration file via the new 
    ``REQUIRE_AUX_PRODUCTS`` parameter. The auxiliary products are specified as a white-space separated list,
    e.g. ``REQUIRE_AUX_PRODUCTS = DST VZN AOD``. Custom products may also be specified (*Int16!*), thus you can invent 
    and use new tags. An auxiliary product is a product should always accompany the main product (usually ``BOA``).
    In the UDF, the auxiliary products are appended to the data array, thus increasing the number of bands.
    The bandnames of the auxiliary products are set to the product name, e.g. ``DST`` for the DST product.
    If no auxiliary products are wanted, the user can set ``REQUIRE_AUX_PRODUCTS = NULL``.

  - in ``force-higher-level``, UDF sub-module:
    the ``REQUIRE_AUX_PRODUCTS`` mechanism has been implemented here as well. 
    You may use the ``DST``, ``HOT``, and ``VZN`` products.
    Before, the usage of a specific product was solely managed by using a corresponding score > 0. 
    To prevent accidental use of a product, the user must now explicitly specify the product in addition to the score.

